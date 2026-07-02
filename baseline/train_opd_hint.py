"""Entry point for Grounding-Hint Distillation (GHD).

Standalone counterpart to ``baseline/train_opd.py``. Trains a (full-FT by default)
student against a separate, frozen, stronger same-family VLM teacher using
on-policy reverse-KL distillation, where the teacher — and only the teacher — is
shown the GT evidence bounding box as a text coordinate hint
(:class:`baseline.hint.opd_hint_collator.OPDHintDataCollator` +
:class:`baseline.hint.opd_hint_trainer.OPDHintTrainer`). ViGOS / vanilla OPD code
paths are untouched; this reuses both as libraries.

The dataset must carry an evidence-box column (``--bbox_field``, default ``bbox``,
a string ``"[x1,y1,x2,y2]"`` normalized to [0,1] — the saliency-r1-8k schema).

Example
-------
    MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-2B-Instruct \\
    TEACHER_MODEL=<stronger same-family VLM> \\
    DATASET_NAME=peterant330/saliency-r1-8k ANSWER_FIELD=solution \\
    bash scripts/train_opd_hint_qwen3_2b.sh
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    AutoProcessor,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

import vigos.dataset_utils as dataset_utils
from baseline.hint.opd_hint_collator import HINT_TEMPLATE, OPDHintDataCollator
from baseline.hint.opd_hint_trainer import OPDHintTrainer
from baseline.opd_data_collator import resolve_opd_system_prompt
from baseline.opd_dataset import load_opd_dataset
from baseline.probe.saliency_data import parse_bbox_norm
from baseline.train_opd import (
    OPDScriptArguments,
    _OPDWandBConfigCallback,
    _chat_template_kwargs_from_args,
    _opd_model_class_for_checkpoint,
)
from vigos.train_vigos import (
    DEFAULT_LEARNING_RATE,
    _cli_arg_was_provided,
    _dtype,
    _reporting_to_wandb,
)

_load_dataset = load_opd_dataset
_filter_tiny_image_samples = dataset_utils.filter_tiny_image_samples


@dataclass
class OPDHintScriptArguments(OPDScriptArguments):
    """Vanilla OPD knobs (inherited) + grounding-privilege knobs."""

    # How the teacher is privileged with the box: "hint" = full image + text coords
    # (direction); "crop" = image cropped to the box, no text (zoom).
    teacher_privilege_mode: str = "hint"
    # Dataset column holding the GT evidence box (string "[x1,y1,x2,y2]" in [0,1],
    # or a list/tuple; parse_bbox_norm order-normalizes / clamps / drops degenerate).
    bbox_field: str = "bbox"
    # Drop rows whose bbox does not parse (default on): every kept row then gives the
    # teacher a real privileged signal. Set false to keep all rows (no-bbox rows fall
    # back to the plain prompt => vanilla OPD for those rows).
    filter_no_bbox: bool = True
    # hint mode: the hint sentence ('{bbox}' is filled per sample). Override for ablations.
    hint_template: str = HINT_TEMPLATE
    hint_coord_decimals: int = 2
    # crop mode: context padding around the box (fraction of box w/h per side).
    crop_padding: float = 0.0


def _filter_samples_without_bbox(dataset, bbox_field: str):
    if bbox_field not in dataset.column_names:
        raise ValueError(
            f"--bbox_field {bbox_field!r} is not a dataset column "
            f"(columns: {dataset.column_names}). Grounding-Hint Distillation needs GT "
            "evidence boxes: point --dataset_name at a bbox-carrying dataset (e.g. "
            "peterant330/saliency-r1-8k), set --bbox_field, or pass --filter_no_bbox "
            "false to train hint-free (== vanilla OPD)."
        )
    return dataset.filter(
        lambda value: parse_bbox_norm(value) is not None,
        input_columns=[bbox_field],
        desc=f"Filtering rows without a parseable '{bbox_field}' evidence box",
    )


def main() -> None:
    parser = HfArgumentParser((OPDHintScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    chat_template_kwargs = _chat_template_kwargs_from_args(script_args)
    if not _cli_arg_was_provided("--learning_rate", "--learning-rate"):
        training_args.learning_rate = DEFAULT_LEARNING_RATE

    if not script_args.dataset_name:
        raise ValueError("--dataset_name is required (a HuggingFace dataset id or dir).")
    if script_args.teacher_source != "local_hf":
        raise ValueError(
            "Grounding-Hint Distillation requires teacher_source='local_hf' (the "
            "privileged prompt is built locally and reverse KL needs full teacher logits)."
        )
    if not script_args.teacher_model_name_or_path:
        raise ValueError("--teacher_model_name_or_path is required for local_hf.")

    if script_args.run_config:
        lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        eff = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * world_size
        )
        training_args.run_name = f"{script_args.run_config}_lr{lr_str}_bs{eff}"
        if not Path(training_args.output_dir).name == script_args.run_config:
            training_args.output_dir = str(
                Path(training_args.output_dir) / script_args.run_config
            )
    elif not training_args.run_name or training_args.run_name == training_args.output_dir:
        training_args.run_name = (
            os.path.basename(os.path.normpath(training_args.output_dir)) or "opd_hint"
        )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("\n" + "=" * 80)
        print("GROUNDING-HINT DISTILLATION (GHD) RUN CONFIGURATION")
        print("=" * 80)
        print(f"Student model: {script_args.model_name_or_path}")
        print(f"Teacher model: {script_args.teacher_model_name_or_path}")
        print(f"Finetuning mode: {script_args.finetuning_mode}")
        print(f"Dataset name: {script_args.dataset_name}")
        print(f"Answer/reference field: {script_args.answer_field}")
        print(
            f"Teacher privilege mode: {script_args.teacher_privilege_mode!r} "
            f"({'cropped evidence image (zoom)' if script_args.teacher_privilege_mode == 'crop' else 'text coordinate hint (direction)'})"
        )
        print(
            f"Bbox field: {script_args.bbox_field!r}  "
            f"filter_no_bbox={script_args.filter_no_bbox}  "
            f"coord_decimals={script_args.hint_coord_decimals}  "
            f"crop_padding={script_args.crop_padding}"
        )
        if script_args.teacher_privilege_mode == "hint":
            print(f"Hint template: {script_args.hint_template!r}")
        print(
            f"OPD system prompt: style={script_args.opd_system_prompt!r} -> "
            f"{resolve_opd_system_prompt(script_args.opd_system_prompt)!r}"
        )
        print(f"Chat template kwargs: {chat_template_kwargs}")
        print(
            "Distillation: "
            f"loss_mode={script_args.opd_loss_mode}, "
            f"kl_direction={script_args.opd_kl_direction}, "
            f"top_k={script_args.opd_top_k}, lambda_opd={script_args.lambda_opd}, "
            f"distill_temperature={script_args.distill_temperature}"
        )
        print(f"Output directory: {training_args.output_dir}")
        print("=" * 80 + "\n")

    set_seed(training_args.seed)

    processor_kwargs = {
        "trust_remote_code": script_args.trust_remote_code,
        "use_fast": False,
    }
    if script_args.max_pixels is not None:
        processor_kwargs["max_pixels"] = script_args.max_pixels
    if script_args.min_pixels is not None:
        processor_kwargs["min_pixels"] = script_args.min_pixels
    processor = AutoProcessor.from_pretrained(
        script_args.model_name_or_path,
        **processor_kwargs,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Coalesce an unmapped (padded-vocab) id to "" so one stray rollout token can't
    # kill the slow tokenizer mid-decode (same guard as vanilla OPD).
    if hasattr(tokenizer, "_convert_id_to_token"):
        _orig_id_to_token = tokenizer._convert_id_to_token

        def _safe_id_to_token(index, _orig=_orig_id_to_token):
            token = _orig(index)
            return "" if token is None else token

        tokenizer._convert_id_to_token = _safe_id_to_token

    # --- Student ----------------------------------------------------------------
    model_class, model_type = _opd_model_class_for_checkpoint(
        script_args.model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
    )
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"Resolved student model_type={model_type} to {model_class.__name__}.")
    model = model_class.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
        attn_implementation=script_args.attn_implementation,
        dtype=_dtype(script_args.torch_dtype),
    )
    model.config.use_cache = False if training_args.gradient_checkpointing else True
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if script_args.finetuning_mode == "full":
        if script_args.freeze_vision_tower:
            frozen = 0
            for name, param in model.named_parameters():
                if "visual." in name:
                    param.requires_grad_(False)
                    frozen += param.numel()
            if os.environ.get("LOCAL_RANK", "0") == "0":
                print(f"Froze vision tower ('visual.*'): {frozen / 1e6:.1f}M params.")
        if os.environ.get("LOCAL_RANK", "0") == "0":
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"Full fine-tuning: {trainable / 1e9:.2f}B trainable / "
                f"{total / 1e9:.2f}B total parameters."
            )
    elif script_args.finetuning_mode == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        raw_targets = script_args.lora_target_modules.strip()
        target_modules = (
            "all-linear"
            if raw_targets == "all-linear"
            else [m.strip() for m in raw_targets.split(",") if m.strip()]
        )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=script_args.lora_r,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                target_modules=target_modules,
                bias="none",
            ),
        )
        model.print_trainable_parameters()
    else:
        raise ValueError(
            f"Unknown finetuning_mode {script_args.finetuning_mode!r}; "
            "expected 'full' or 'lora'."
        )

    # --- Teacher (local_hf, frozen) --------------------------------------------
    teacher_class, teacher_type = _opd_model_class_for_checkpoint(
        script_args.teacher_model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
    )
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(
            f"Loading GHD teacher {script_args.teacher_model_name_or_path} "
            f"(model_type={teacher_type} -> {teacher_class.__name__})."
        )
    teacher_model = teacher_class.from_pretrained(
        script_args.teacher_model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
        attn_implementation=script_args.teacher_attn_implementation,
        dtype=_dtype(script_args.teacher_torch_dtype),
    )
    teacher_model.config.use_cache = False
    teacher_model.requires_grad_(False)
    teacher_model.eval()

    # --- Data -------------------------------------------------------------------
    dataset = _load_dataset(script_args.dataset_name, script_args.dataset_split)
    if script_args.filter_tiny_images:
        pre = len(dataset)
        dataset = _filter_tiny_image_samples(
            dataset, min_image_size=script_args.min_image_size
        )
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"Tiny-image filter removed {pre - len(dataset)}/{pre} samples.")
    if script_args.filter_no_bbox:
        pre = len(dataset)
        dataset = _filter_samples_without_bbox(dataset, script_args.bbox_field)
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(
                f"Bbox filter kept {len(dataset)}/{pre} samples carrying a parseable "
                f"'{script_args.bbox_field}' evidence box."
            )
    # Cap LAST so a smoke-test --max_train_samples yields exactly N usable (boxed) rows.
    if script_args.max_train_samples is not None:
        dataset = dataset.select(range(min(script_args.max_train_samples, len(dataset))))

    data_collator = OPDHintDataCollator(
        processor=processor,
        max_prompt_length=script_args.max_prompt_length,
        answer_field=script_args.answer_field,
        opd_prompt_suffix=script_args.opd_prompt_suffix,
        system_prompt=resolve_opd_system_prompt(script_args.opd_system_prompt),
        teacher_privilege_mode=script_args.teacher_privilege_mode,
        bbox_field=script_args.bbox_field,
        hint_template=script_args.hint_template,
        hint_coord_decimals=script_args.hint_coord_decimals,
        crop_padding=script_args.crop_padding,
        chat_template_kwargs=chat_template_kwargs,
    )

    # --- Trainer ----------------------------------------------------------------
    trainer = OPDHintTrainer(
        model=model,
        args=training_args,
        model_name_or_path=script_args.model_name_or_path,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=processor,
        processor=processor,
        teacher_model=teacher_model,
        teacher_source="local_hf",
        lambda_opd=script_args.lambda_opd,
        opd_loss_mode=script_args.opd_loss_mode,
        opd_kl_direction=script_args.opd_kl_direction,
        opd_top_k=script_args.opd_top_k,
        max_prompt_length=script_args.max_prompt_length,
        max_completion_length=script_args.max_completion_length,
        generation_temperature=script_args.generation_temperature,
        generation_top_p=script_args.generation_top_p,
        generation_top_k=script_args.generation_top_k,
        distill_temperature=script_args.distill_temperature,
        token_loss_clip=(
            script_args.token_loss_clip if script_args.token_loss_clip > 0 else None
        ),
        presence_penalty=script_args.presence_penalty,
        repetition_penalty=script_args.repetition_penalty,
        min_p=script_args.min_p,
        use_vllm=script_args.use_vllm,
        vllm_mode=script_args.vllm_mode,
        vllm_gpu_memory_utilization=script_args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=script_args.vllm_tensor_parallel_size,
        vllm_sync_frequency=script_args.vllm_sync_frequency,
        vllm_max_model_len=script_args.vllm_max_model_len
        or script_args.max_prompt_length + script_args.max_completion_length,
        vllm_max_num_seqs=script_args.vllm_max_num_seqs,
        vllm_disable_custom_all_reduce=script_args.vllm_disable_custom_all_reduce,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        completion_log_steps=script_args.completion_log_steps,
        completion_log_max_samples=script_args.completion_log_max_samples,
    )

    if _reporting_to_wandb(training_args):
        trainer.add_callback(
            _OPDWandBConfigCallback(
                {
                    "opd_method": f"ghd_{script_args.teacher_privilege_mode}",
                    "opd_finetuning_mode": script_args.finetuning_mode,
                    "opd_student_model": script_args.model_name_or_path,
                    "opd_teacher_model": script_args.teacher_model_name_or_path,
                    "opd_dataset_name": script_args.dataset_name,
                    "opd_train_dataset_size": len(dataset),
                    "opd_lambda_opd": script_args.lambda_opd,
                    "opd_loss_mode": script_args.opd_loss_mode,
                    "opd_kl_direction": script_args.opd_kl_direction,
                    "opd_top_k": script_args.opd_top_k,
                    "ghd_teacher_privilege_mode": script_args.teacher_privilege_mode,
                    "ghd_bbox_field": script_args.bbox_field,
                    "ghd_filter_no_bbox": script_args.filter_no_bbox,
                    "ghd_hint_template": script_args.hint_template,
                    "ghd_hint_coord_decimals": script_args.hint_coord_decimals,
                    "ghd_crop_padding": script_args.crop_padding,
                    "opd_chat_template_kwargs": chat_template_kwargs,
                    "opd_enable_thinking": script_args.enable_thinking,
                }
            )
        )

    if trainer.accelerator.is_main_process:
        num_proc = trainer.accelerator.num_processes
        eff = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * num_proc
        )
        print(
            f"[GHD] num_processes(world_size)={num_proc}  "
            f"per_device_bs={training_args.per_device_train_batch_size}  "
            f"grad_accum={training_args.gradient_accumulation_steps}  "
            f"-> effective_batch={eff}  train_size={len(dataset)}",
            flush=True,
        )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
