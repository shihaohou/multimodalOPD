"""Entry point for Locate-Once Grounding (LOG).

Standalone counterpart to ``baseline/train_opd_hint.py``. Trains a (full-FT by
default) student with the verified hidden-hint OPD spine PLUS an explicit,
student-generated evidence box trained by RL:

* the **student** rolls out from a *locate-once* prompt — it opens its ``<think>``
  with one ``<box>[x1,y1,x2,y2]</box>`` (no crop), then reasons and answers;
* the **teacher** (frozen, stronger, same-family) is scored on the *hidden-hint*
  prompt (silently handed the GT box, forbidden from verbalizing it) — the OPD target;
* the **OPD** reverse-KL covers the answer/reasoning span with the box span masked
  out; the **RL** (GRPO, group-normalized IoU reward gated by answer correctness)
  reinforces the box coordinate tokens.

ViGOS / vanilla-OPD / GHD code paths are reused as libraries, untouched. The dataset
must carry a GT evidence box (``--bbox_field``, default ``bbox``; saliency-r1-8k or
Visual-CoT) — RL needs it for the IoU reward.

Example
-------
    MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-2B-Instruct \\
    TEACHER_MODEL=<stronger same-family VLM> \\
    DATASET_NAME=peterant330/saliency-r1-8k ANSWER_FIELD=solution \\
    bash scripts/train_opd_locate_qwen3_2b.sh
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
from baseline.locate.opd_locate_collator import LOCATE_SYSTEM_PROMPT, OPDLocateDataCollator
from baseline.locate.opd_locate_trainer import OPDLocateTrainer
from baseline.opd_data_collator import resolve_opd_system_prompt
from baseline.opd_dataset import load_opd_dataset
from baseline.train_opd import (
    _OPDWandBConfigCallback,
    _opd_model_class_for_checkpoint,
)
from baseline.train_opd_hint import OPDHintScriptArguments, _filter_samples_without_bbox
from vigos.train_vigos import (
    DEFAULT_LEARNING_RATE,
    _cli_arg_was_provided,
    _dtype,
    _reporting_to_wandb,
)

_load_dataset = load_opd_dataset
_filter_tiny_image_samples = dataset_utils.filter_tiny_image_samples


@dataclass
class OPDLocateScriptArguments(OPDHintScriptArguments):
    """GHD knobs (inherited) + the locate-once / box-RL knobs."""

    # Rollouts sampled per prompt for the GRPO baseline (the collator expands each
    # prompt into a contiguous group of this many rows). With per_device_train_batch_size
    # counting PROMPTS, effective rollouts/step = per_device_bs * group_size * grad_accum
    # * world_size; set per_device_bs small (1-2) and let group_size do the batching.
    group_size: int = 8
    # Loss weights. lambda_opd (inherited) weights the hidden-hint OPD KL; lambda_rl
    # weights the box PG term. Doc start point: 1.0 / 0.5.
    lambda_rl: float = 0.5
    # RL reward: "gated_iou" = IoU(student_box, GT_box) only when the answer is correct
    # (DeepEyes conditional tool reward, the default); "iou" = ungated (ablation).
    rl_reward: str = "gated_iou"
    # Warmup (gated_iou only): add `rl_ungated_weight * IoU` so a well-placed box still
    # earns a small signal while answer accuracy is low. 0.0 = pure gated.
    rl_ungated_weight: float = 0.0
    # GRPO advantage normalization: True = (r - mean)/(std + eps); False = r - mean (Dr.GRPO).
    rl_normalize_adv: bool = True
    rl_adv_eps: float = 1e-6
    # Deferred position gate (off): apply the OPD KL only where teacher logprob >
    # student logprob on the sampled token. Available for ablation once the spine works.
    kl_position_gate: bool = False
    # Student locate-once system prompt (must contain "<box>"). Override for ablations.
    locate_system_prompt: str = LOCATE_SYSTEM_PROMPT
    # OPD teacher hint = inherited no-verbalize HINT_TEMPLATE. We do NOT let the teacher state
    # the box: CapCurriculum verbalizes coordinates in its reasoning, and reverse-KL then
    # forces the student to reproduce unknowable coords in the (unmasked) reasoning span ->
    # token-salad collapse (de5e4c5). Cold-start already taught the student to emit boxes, and
    # its <box> span is masked from OPD, so the teacher needn't (and must not) verbalize it.


def main() -> None:
    parser = HfArgumentParser((OPDLocateScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    if not _cli_arg_was_provided("--learning_rate", "--learning-rate"):
        training_args.learning_rate = DEFAULT_LEARNING_RATE

    if not script_args.dataset_name:
        raise ValueError("--dataset_name is required (a HuggingFace dataset id or dir).")
    if script_args.teacher_source != "local_hf":
        raise ValueError(
            "Locate-Once Grounding requires teacher_source='local_hf' (the hidden-hint "
            "prompt is built locally and reverse KL needs full teacher logits)."
        )
    if not script_args.teacher_model_name_or_path:
        raise ValueError("--teacher_model_name_or_path is required for local_hf.")
    if script_args.teacher_privilege_mode != "hint":
        raise ValueError(
            "Locate-Once Grounding uses the HIDDEN-HINT teacher "
            "(--teacher_privilege_mode hint); 'crop' is not part of this fork."
        )
    if "<box>" not in script_args.locate_system_prompt:
        raise ValueError(
            "--locate_system_prompt must instruct the student to emit a <box>...</box> "
            "(it is the RL handle)."
        )

    if script_args.run_config:
        lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        eff = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * world_size
            * script_args.group_size
        )
        training_args.run_name = f"{script_args.run_config}_lr{lr_str}_rollouts{eff}"
        if not Path(training_args.output_dir).name == script_args.run_config:
            training_args.output_dir = str(
                Path(training_args.output_dir) / script_args.run_config
            )
    elif not training_args.run_name or training_args.run_name == training_args.output_dir:
        training_args.run_name = (
            os.path.basename(os.path.normpath(training_args.output_dir)) or "opd_locate"
        )

    # OPD teacher = the hidden-hint GHD teacher (plain think + no-verbalize hint): grounded
    # (it silently sees the GT box) but MUST NOT verbalize the box, or it collapses (see the
    # hint_template note above). The student's <box> span is masked from OPD regardless.
    teacher_system_prompt = resolve_opd_system_prompt(script_args.opd_system_prompt)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("\n" + "=" * 80)
        print("LOCATE-ONCE GROUNDING (LOG) RUN CONFIGURATION")
        print("=" * 80)
        print(f"Student model: {script_args.model_name_or_path}")
        print(f"Teacher model: {script_args.teacher_model_name_or_path}")
        print(f"Finetuning mode: {script_args.finetuning_mode}")
        print(f"Dataset name: {script_args.dataset_name}")
        print(f"Answer/reference field: {script_args.answer_field}")
        print(f"Bbox field: {script_args.bbox_field}  filter_no_bbox={script_args.filter_no_bbox}")
        print(f"Student (locate-once) prompt: {script_args.locate_system_prompt!r}")
        print(f"Teacher (hidden-hint, no-verbalize) prompt: {teacher_system_prompt!r}")
        print(f"Hint template: {script_args.hint_template!r}")
        print(
            "Distillation (OPD): "
            f"loss_mode={script_args.opd_loss_mode}, kl_direction={script_args.opd_kl_direction}, "
            f"top_k={script_args.opd_top_k}, lambda_opd={script_args.lambda_opd}, "
            f"distill_temperature={script_args.distill_temperature}, "
            f"kl_position_gate={script_args.kl_position_gate}"
        )
        print(
            "Box RL (GRPO): "
            f"group_size={script_args.group_size}, lambda_rl={script_args.lambda_rl}, "
            f"reward={script_args.rl_reward}, ungated_weight={script_args.rl_ungated_weight}, "
            f"normalize_adv={script_args.rl_normalize_adv}"
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
            f"Loading LOG teacher {script_args.teacher_model_name_or_path} "
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
                f"'{script_args.bbox_field}' evidence box (RL needs it for the IoU reward)."
            )
    if script_args.max_train_samples is not None:
        dataset = dataset.select(range(min(script_args.max_train_samples, len(dataset))))

    data_collator = OPDLocateDataCollator(
        processor=processor,
        max_prompt_length=script_args.max_prompt_length,
        answer_field=script_args.answer_field,
        opd_prompt_suffix=script_args.opd_prompt_suffix,
        # Student gets the locate-once prompt; the teacher keeps the plain think prompt
        # (+ the silent hint), decoupled via teacher_system_prompt.
        system_prompt=script_args.locate_system_prompt,
        teacher_system_prompt=teacher_system_prompt,
        teacher_privilege_mode="hint",
        bbox_field=script_args.bbox_field,
        hint_template=script_args.hint_template,
        hint_coord_decimals=script_args.hint_coord_decimals,
        group_size=script_args.group_size,
    )

    # --- Trainer ----------------------------------------------------------------
    trainer = OPDLocateTrainer(
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
        lambda_rl=script_args.lambda_rl,
        rl_reward=script_args.rl_reward,
        rl_ungated_weight=script_args.rl_ungated_weight,
        rl_normalize_adv=script_args.rl_normalize_adv,
        rl_adv_eps=script_args.rl_adv_eps,
        group_size=script_args.group_size,
        kl_position_gate=script_args.kl_position_gate,
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
        completion_log_steps=script_args.completion_log_steps,
        completion_log_max_samples=script_args.completion_log_max_samples,
    )

    if _reporting_to_wandb(training_args):
        trainer.add_callback(
            _OPDWandBConfigCallback(
                {
                    "opd_method": "locate_once",
                    "opd_finetuning_mode": script_args.finetuning_mode,
                    "opd_student_model": script_args.model_name_or_path,
                    "opd_teacher_model": script_args.teacher_model_name_or_path,
                    "opd_dataset_name": script_args.dataset_name,
                    "opd_train_dataset_size": len(dataset),
                    "opd_lambda_opd": script_args.lambda_opd,
                    "opd_loss_mode": script_args.opd_loss_mode,
                    "opd_kl_direction": script_args.opd_kl_direction,
                    "opd_top_k": script_args.opd_top_k,
                    "locate_group_size": script_args.group_size,
                    "locate_lambda_rl": script_args.lambda_rl,
                    "locate_rl_reward": script_args.rl_reward,
                    "locate_rl_ungated_weight": script_args.rl_ungated_weight,
                    "locate_rl_normalize_adv": script_args.rl_normalize_adv,
                    "locate_kl_position_gate": script_args.kl_position_gate,
                    "locate_bbox_field": script_args.bbox_field,
                    "locate_hint_template": script_args.hint_template,
                }
            )
        )

    if trainer.accelerator.is_main_process:
        num_proc = trainer.accelerator.num_processes
        eff = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * num_proc
            * script_args.group_size
        )
        print(
            f"[LOG] num_processes(world_size)={num_proc}  "
            f"per_device_prompts={training_args.per_device_train_batch_size}  "
            f"group_size={script_args.group_size}  "
            f"grad_accum={training_args.gradient_accumulation_steps}  "
            f"-> effective_rollouts/step={eff}  train_size={len(dataset)}",
            flush=True,
        )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
