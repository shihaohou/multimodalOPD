"""Entry point for OPD + differentiable evidence-alignment training (Step 3).

Mirrors ``baseline/train_opd.py`` but swaps in
:class:`baseline.evidence.opd_evidence_trainer.OPDEvidenceTrainer` and adds the
``--evidence_*`` / ``--lambda_evidence`` knobs. Vanilla OPD and ViGOS entry points
are left untouched. Requires ``teacher_source='local_hf'`` (the evidence term
needs a full local teacher forward).

Example
-------
    DATASET_NAME=LMMs-Lab-Turtle/Vision-SR1-47K \\
    TEACHER_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \\
    bash scripts/train_opd_evidence_qwen25_3b.sh
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

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, HfArgumentParser, TrainingArguments, set_seed

import vigos.dataset_utils as dataset_utils
from baseline.evidence.opd_evidence_trainer import OPDEvidenceTrainer
from baseline.opd_data_collator import OPDDataCollator
from baseline.train_opd import OPDScriptArguments, _OPDWandBConfigCallback
from vigos.train_vigos import (
    DEFAULT_LEARNING_RATE,
    _cli_arg_was_provided,
    _dtype,
    _model_class_for_checkpoint,
    _reporting_to_wandb,
)


@dataclass
class OPDEvidenceScriptArguments(OPDScriptArguments):
    """OPD knobs (inherited) + evidence-alignment knobs."""

    lambda_evidence: float = 1.0
    evidence_max_samples: int = 1
    # Comma list of decoder layers to sum saliency over (default: all layers).
    evidence_layers: str | None = None
    evidence_top_ratio: float = 0.2
    evidence_min_tokens: int = 1
    evidence_max_tokens: int = 8
    evidence_signed: bool = True
    evidence_kl_direction: str = "forward"  # token-selection / kl-gate direction
    evidence_gate_temp: float = 1.0
    evidence_gate_h0: float = 0.9
    evidence_gate_tau: float = 0.1
    evidence_kl_threshold: float = 0.0
    evidence_mass_threshold: float = 0.0
    # Cap image resolution => caps #visual tokens => caps the eager-attention S^2
    # memory (the OOM lever for evidence). 0 = use the processor default (which can
    # be ~12.8M px / thousands of visual tokens -> OOM under output_attentions).
    max_pixels: int = 0
    min_pixels: int = 0


def main() -> None:
    parser = HfArgumentParser((OPDEvidenceScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    if not _cli_arg_was_provided("--learning_rate", "--learning-rate"):
        training_args.learning_rate = DEFAULT_LEARNING_RATE

    if not script_args.dataset_name:
        raise ValueError("--dataset_name is required (a HuggingFace dataset id).")
    if script_args.teacher_source != "local_hf":
        raise ValueError(
            "OPD evidence training requires teacher_source='local_hf' "
            "(the evidence term needs a full local teacher forward)."
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
            os.path.basename(os.path.normpath(training_args.output_dir)) or "opd_evidence"
        )

    evidence_layers = (
        tuple(int(x) for x in script_args.evidence_layers.split(",") if x.strip())
        if script_args.evidence_layers
        else None
    )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("\n" + "=" * 80)
        print("OPD + EVIDENCE-ALIGNMENT RUN CONFIGURATION")
        print("=" * 80)
        print(f"Student model: {script_args.model_name_or_path}")
        print(f"Teacher model: {script_args.teacher_model_name_or_path}")
        print(f"Finetuning mode: {script_args.finetuning_mode}")
        print(
            "OPD: "
            f"loss_mode={script_args.opd_loss_mode}, "
            f"kl_direction={script_args.opd_kl_direction}, "
            f"lambda_opd={script_args.lambda_opd}"
        )
        print(
            "Evidence: "
            f"lambda_evidence={script_args.lambda_evidence}, "
            f"max_samples={script_args.evidence_max_samples}, "
            f"layers={evidence_layers or 'all'}, "
            f"top_ratio={script_args.evidence_top_ratio}, "
            f"signed={script_args.evidence_signed}, "
            f"gate(h0={script_args.evidence_gate_h0},tau={script_args.evidence_gate_tau})"
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

    # Cap image resolution to bound the eager-attention memory (evidence OOM lever).
    if script_args.max_pixels > 0 or script_args.min_pixels > 0:
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            size = getattr(image_processor, "size", None)
            if script_args.max_pixels > 0:
                image_processor.max_pixels = script_args.max_pixels
                if isinstance(size, dict):
                    size["longest_edge"] = script_args.max_pixels
            if script_args.min_pixels > 0:
                image_processor.min_pixels = script_args.min_pixels
                if isinstance(size, dict):
                    size["shortest_edge"] = script_args.min_pixels
            if os.environ.get("LOCAL_RANK", "0") == "0":
                print(
                    f"Capped image_processor: min_pixels={getattr(image_processor, 'min_pixels', None)} "
                    f"max_pixels={getattr(image_processor, 'max_pixels', None)} size={size}"
                )

    # --- Student ----------------------------------------------------------------
    model_class, _ = _model_class_for_checkpoint(
        script_args.model_name_or_path, trust_remote_code=script_args.trust_remote_code
    )
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
            for name, param in model.named_parameters():
                if "visual." in name:
                    param.requires_grad_(False)
    elif script_args.finetuning_mode == "lora":
        target_modules = [
            m.strip() for m in script_args.lora_target_modules.split(",") if m.strip()
        ]
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
        raise ValueError(f"Unknown finetuning_mode {script_args.finetuning_mode!r}.")

    # --- Teacher (local_hf, frozen) --------------------------------------------
    teacher_class, _ = _model_class_for_checkpoint(
        script_args.teacher_model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
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
    dataset = dataset_utils.load_vigos_dataset(
        script_args.dataset_name, script_args.dataset_split
    )
    if script_args.max_train_samples is not None:
        dataset = dataset.select(
            range(min(script_args.max_train_samples, len(dataset)))
        )
    if script_args.filter_tiny_images:
        dataset = dataset_utils.filter_tiny_image_samples(
            dataset, min_image_size=script_args.min_image_size
        )

    data_collator = OPDDataCollator(
        processor=processor,
        max_prompt_length=script_args.max_prompt_length,
        answer_field=script_args.answer_field,
        opd_prompt_suffix=script_args.opd_prompt_suffix,
    )

    # --- Trainer ----------------------------------------------------------------
    trainer = OPDEvidenceTrainer(
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
        # evidence knobs
        lambda_evidence=script_args.lambda_evidence,
        evidence_max_samples=script_args.evidence_max_samples,
        evidence_layers=evidence_layers,
        evidence_top_ratio=script_args.evidence_top_ratio,
        evidence_min_tokens=script_args.evidence_min_tokens,
        evidence_max_tokens=script_args.evidence_max_tokens,
        evidence_signed=script_args.evidence_signed,
        evidence_kl_direction=script_args.evidence_kl_direction,
        evidence_gate_temp=script_args.evidence_gate_temp,
        evidence_gate_h0=script_args.evidence_gate_h0,
        evidence_gate_tau=script_args.evidence_gate_tau,
        evidence_kl_threshold=script_args.evidence_kl_threshold,
        evidence_mass_threshold=script_args.evidence_mass_threshold,
        # generation / vllm / loss plumbing (same as vanilla OPD)
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
                    "opd_method": "opd_evidence",
                    "opd_student_model": script_args.model_name_or_path,
                    "opd_teacher_model": script_args.teacher_model_name_or_path,
                    "opd_dataset_name": script_args.dataset_name,
                    "opd_lambda_opd": script_args.lambda_opd,
                    "evidence_lambda": script_args.lambda_evidence,
                    "evidence_max_samples": script_args.evidence_max_samples,
                    "evidence_layers": evidence_layers,
                    "evidence_top_ratio": script_args.evidence_top_ratio,
                    "evidence_signed": script_args.evidence_signed,
                    "evidence_kl_direction": script_args.evidence_kl_direction,
                    "evidence_gate_h0": script_args.evidence_gate_h0,
                    "evidence_gate_tau": script_args.evidence_gate_tau,
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
            f"[OPD-evidence] num_processes(world_size)={num_proc}  "
            f"per_device_bs={training_args.per_device_train_batch_size}  "
            f"grad_accum={training_args.gradient_accumulation_steps}  "
            f"-> effective_batch={eff}  lambda_evidence={script_args.lambda_evidence}",
            flush=True,
        )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
