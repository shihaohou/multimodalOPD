"""Entry point for Visual-Value Evidence Slot OPD.

The student prompt contains image + question + ``<EVID>`` slots. The teacher
prompt contains image + question + hidden evidence-box hint + the same slots. OPD
KL is computed on the student's on-policy rollout, while the auxiliary loss aligns
a structured latent:

    slot query -> attend over visual token K/V -> visual-value evidence state

The aligned value is therefore pooled from image tokens, not copied from the
teacher's raw text hidden state.
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

from transformers import AutoProcessor, HfArgumentParser, TrainingArguments, set_seed

import vigos.dataset_utils as dataset_utils
from baseline.anchor.opd_anchor_collator import (
    DEFAULT_ANCHOR_ANSWER_CUE,
    DEFAULT_ANCHOR_TOKEN,
    OPDAnchorDataCollator,
    build_anchor_text,
)
from baseline.evidence_slot.opd_evidence_slot_trainer import OPDEvidenceSlotTrainer
from baseline.hint.opd_hint_collator import HINT_TEMPLATE
from baseline.opd_data_collator import resolve_opd_system_prompt
from baseline.opd_dataset import load_opd_dataset
from baseline.probe.saliency_data import parse_bbox_norm
from baseline.train_opd import (
    OPDScriptArguments,
    _OPDWandBConfigCallback,
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
class OPDEvidenceSlotScriptArguments(OPDScriptArguments):
    """Vanilla OPD knobs + hidden-hint privilege + evidence-slot knobs."""

    teacher_privilege_mode: str = "hint"
    bbox_field: str = "bbox"
    filter_no_bbox: bool = True
    hint_template: str = HINT_TEMPLATE
    hint_coord_decimals: int = 2
    crop_padding: float = 0.0

    anchor_token: str = DEFAULT_ANCHOR_TOKEN
    num_anchor_tokens: int = 1
    anchor_indexed_tokens: bool = True
    anchor_answer_cue: str = DEFAULT_ANCHOR_ANSWER_CUE

    lambda_evidence_slot: float = 1.0
    evidence_slot_projection_dim: int = 1024
    evidence_slot_projector_bias: bool = False
    evidence_slot_train_teacher_projector: bool = False
    evidence_slot_bbox_bias_beta: float = 2.0
    evidence_slot_normalize_qk: bool = True
    evidence_slot_inject_student: bool = True
    evidence_slot_injection_scale: float = 1.0
    evidence_slot_injection_match_norm: bool = True
    # Put the hidden hint after <EVID> so the teacher slot query is conditioned
    # only on image+question; the completion still sees the hint for OPD scoring.
    evidence_slot_hint_after_anchor: bool = True


def _filter_samples_without_bbox(dataset, bbox_field: str):
    if bbox_field not in dataset.column_names:
        raise ValueError(
            f"--bbox_field {bbox_field!r} is not a dataset column "
            f"(columns: {dataset.column_names}). Visual-Value Evidence Slot OPD "
            "needs GT evidence boxes for the hidden-hint teacher. Set --bbox_field "
            "to the right column, or pass --filter_no_bbox false to fall back to "
            "plain teacher prompts on rows without boxes."
        )
    return dataset.filter(
        lambda value: parse_bbox_norm(value) is not None,
        input_columns=[bbox_field],
        desc=f"Filtering rows without a parseable '{bbox_field}' evidence box",
    )


def main() -> None:
    parser = HfArgumentParser((OPDEvidenceSlotScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    if not _cli_arg_was_provided("--learning_rate", "--learning-rate"):
        training_args.learning_rate = DEFAULT_LEARNING_RATE

    if not script_args.dataset_name:
        raise ValueError("--dataset_name is required (a HuggingFace dataset id or dir).")
    if script_args.teacher_source != "local_hf":
        raise ValueError(
            "Visual-Value Evidence Slot OPD requires teacher_source='local_hf' "
            "because the evidence-slot loss needs teacher hidden states."
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
            os.path.basename(os.path.normpath(training_args.output_dir))
            or "opd_evidence_slot"
        )

    anchor_text = build_anchor_text(
        anchor_token=script_args.anchor_token,
        num_anchor_tokens=script_args.num_anchor_tokens,
        indexed_tokens=script_args.anchor_indexed_tokens,
    )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("\n" + "=" * 80)
        print("VISUAL-VALUE EVIDENCE SLOT OPD RUN CONFIGURATION")
        print("=" * 80)
        print(f"Student model: {script_args.model_name_or_path}")
        print(f"Teacher model: {script_args.teacher_model_name_or_path}")
        print(f"Finetuning mode: {script_args.finetuning_mode}")
        print(f"Dataset name: {script_args.dataset_name}")
        print(f"Answer/reference field: {script_args.answer_field}")
        print(
            f"Teacher privilege mode: {script_args.teacher_privilege_mode!r}  "
            f"bbox_field={script_args.bbox_field!r}  "
            f"filter_no_bbox={script_args.filter_no_bbox}"
        )
        if script_args.teacher_privilege_mode == "hint":
            print(f"Hint template: {script_args.hint_template!r}")
        print(
            f"OPD system prompt: style={script_args.opd_system_prompt!r} -> "
            f"{resolve_opd_system_prompt(script_args.opd_system_prompt)!r}"
        )
        print(
            "Evidence slot: "
            f"anchor_text={anchor_text!r}, "
            f"lambda={script_args.lambda_evidence_slot}, "
            f"projection_dim={script_args.evidence_slot_projection_dim}, "
            f"bbox_beta={script_args.evidence_slot_bbox_bias_beta}, "
            f"normalize_qk={script_args.evidence_slot_normalize_qk}, "
            f"inject_student={script_args.evidence_slot_inject_student}, "
            f"injection_scale={script_args.evidence_slot_injection_scale}, "
            f"match_norm={script_args.evidence_slot_injection_match_norm}, "
            f"hint_after_anchor={script_args.evidence_slot_hint_after_anchor}"
        )
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

    processor = AutoProcessor.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
        use_fast=False,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if hasattr(tokenizer, "_convert_id_to_token"):
        _orig_id_to_token = tokenizer._convert_id_to_token

        def _safe_id_to_token(index, _orig=_orig_id_to_token):
            token = _orig(index)
            return "" if token is None else token

        tokenizer._convert_id_to_token = _safe_id_to_token

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

    teacher_class, teacher_type = _opd_model_class_for_checkpoint(
        script_args.teacher_model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
    )
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(
            f"Loading evidence-slot teacher {script_args.teacher_model_name_or_path} "
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
    if script_args.max_train_samples is not None:
        dataset = dataset.select(range(min(script_args.max_train_samples, len(dataset))))

    data_collator = OPDAnchorDataCollator(
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
        anchor_token=script_args.anchor_token,
        num_anchor_tokens=script_args.num_anchor_tokens,
        anchor_indexed_tokens=script_args.anchor_indexed_tokens,
        anchor_answer_cue=script_args.anchor_answer_cue,
        hint_after_anchor=script_args.evidence_slot_hint_after_anchor,
    )

    trainer = OPDEvidenceSlotTrainer(
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
        lambda_evidence_slot=script_args.lambda_evidence_slot,
        evidence_slot_projection_dim=script_args.evidence_slot_projection_dim,
        evidence_slot_projector_bias=script_args.evidence_slot_projector_bias,
        evidence_slot_train_teacher_projector=(
            script_args.evidence_slot_train_teacher_projector
        ),
        evidence_slot_bbox_bias_beta=script_args.evidence_slot_bbox_bias_beta,
        evidence_slot_normalize_qk=script_args.evidence_slot_normalize_qk,
        evidence_slot_inject_student=script_args.evidence_slot_inject_student,
        evidence_slot_injection_scale=script_args.evidence_slot_injection_scale,
        evidence_slot_injection_match_norm=script_args.evidence_slot_injection_match_norm,
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
        completion_log_steps=script_args.completion_log_steps,
        completion_log_max_samples=script_args.completion_log_max_samples,
    )

    if _reporting_to_wandb(training_args):
        trainer.add_callback(
            _OPDWandBConfigCallback(
                {
                    "opd_method": "visual_value_evidence_slot_opd",
                    "opd_finetuning_mode": script_args.finetuning_mode,
                    "opd_student_model": script_args.model_name_or_path,
                    "opd_teacher_model": script_args.teacher_model_name_or_path,
                    "opd_dataset_name": script_args.dataset_name,
                    "opd_train_dataset_size": len(dataset),
                    "opd_lambda_opd": script_args.lambda_opd,
                    "opd_loss_mode": script_args.opd_loss_mode,
                    "opd_kl_direction": script_args.opd_kl_direction,
                    "opd_top_k": script_args.opd_top_k,
                    "evidence_slot_lambda": script_args.lambda_evidence_slot,
                    "evidence_slot_text": anchor_text,
                    "evidence_slot_projection_dim": (
                        script_args.evidence_slot_projection_dim
                    ),
                    "evidence_slot_bbox_bias_beta": (
                        script_args.evidence_slot_bbox_bias_beta
                    ),
                    "evidence_slot_normalize_qk": script_args.evidence_slot_normalize_qk,
                    "evidence_slot_inject_student": (
                        script_args.evidence_slot_inject_student
                    ),
                    "evidence_slot_injection_scale": (
                        script_args.evidence_slot_injection_scale
                    ),
                    "evidence_slot_injection_match_norm": (
                        script_args.evidence_slot_injection_match_norm
                    ),
                    "evidence_slot_hint_after_anchor": (
                        script_args.evidence_slot_hint_after_anchor
                    ),
                    "ghd_teacher_privilege_mode": script_args.teacher_privilege_mode,
                    "ghd_bbox_field": script_args.bbox_field,
                    "ghd_filter_no_bbox": script_args.filter_no_bbox,
                    "ghd_hint_template": script_args.hint_template,
                    "ghd_hint_coord_decimals": script_args.hint_coord_decimals,
                    "ghd_crop_padding": script_args.crop_padding,
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
            f"[OPD-evidence-slot] num_processes(world_size)={num_proc}  "
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
