"""Entry point for OPD + differentiable TAM visual-evidence alignment training.

Mirrors ``baseline/train_opd.py`` but swaps in
:class:`baseline.tam.tam_trainer.TAMTrainer` and adds the ``--tam_*`` /
``--lambda_tam`` knobs. Vanilla OPD / evidence / ViGOS entry points are left
untouched. Requires ``teacher_source='local_hf'`` (the TAM term needs the
teacher's last-layer hidden states from a local forward).

Example
-------
    DATASET_NAME=LMMs-Lab-Turtle/Vision-SR1-47K \\
    TEACHER_MODEL=Qwen/Qwen3-VL-8B-Instruct \\
    bash scripts/train_opd_tam_qwen3_8b_to_2b.sh
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
from baseline.opd_data_collator import OPDDataCollator
from baseline.tam.tam_trainer import TAMTrainer
from baseline.train_opd import OPDScriptArguments, _OPDWandBConfigCallback
from vigos.train_vigos import (
    DEFAULT_LEARNING_RATE,
    _cli_arg_was_provided,
    _dtype,
    _model_class_for_checkpoint,
    _reporting_to_wandb,
)


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class OPDTAMScriptArguments(OPDScriptArguments):
    """OPD knobs (inherited) + TAM visual-evidence alignment knobs."""

    lambda_tam: float = 1.0
    # Which rollout tokens to align on: "completion" (all, gate selects — default),
    # "answer" (\boxed{} span only), or "reason_answer" (<reason> + \boxed{}).
    tam_align_span: str = "completion"
    tam_use_eci: bool = True
    # Read the map along a detached lm_head row so the gradient lands on the visual
    # representation F^v, not the unembedding (migration doc §3). Keep True.
    tam_detach_lm_head: bool = True
    tam_divergence: str = "cosine"  # "cosine" | "js" | "l1" | "mse"
    tam_blur: bool = True
    # Spatial denoiser on the TAM maps: "gaussian" (fixed blur, default), "rgf" (the
    # paper's Rank-Gaussian Filter — the TAM-MSE-RGF ablation), or "none".
    # tam_blur=false forces "none" (back-compat). The ablation = mse + rgf.
    tam_denoise: str = "gaussian"  # "gaussian" | "rgf" | "none"
    tam_blur_kernel: int = 3
    tam_blur_sigma: float = 1.0
    # Concentration gate (+ mass drop) on/off. false => align ALL aligned tokens with
    # equal weight (the "no gate" first step). True keeps the soft visual-token gate.
    tam_gate: bool = True
    tam_gate_temp: float = 1.0
    tam_gate_h0: float = 0.9
    tam_gate_tau: float = 0.1
    # Hard-drop tokens whose (blurred) teacher map barely responds (teacher grounds
    # nowhere → under Laplace smoothing it is ~uniform, aligning to it is noise). In
    # (0,1) = RELATIVE to the sample's mean teacher mass (portable); >=1 = absolute
    # sum; 0 = keep all (soft gate still down-weights). W&B `tam_mass_kept` = surviving frac.
    tam_mass_threshold: float = 0.0
    # Hard cap on aligned tokens per sample (keep the most concentrated teacher
    # tokens). 0 = no cap (TAM is cheap; the soft gate handles diffuse tokens).
    tam_max_tokens: int = 0
    # Cap image resolution => caps #visual tokens => caps the per-layer hidden-state
    # retention of output_hidden_states. 0 = processor default.
    max_pixels: int = 0
    min_pixels: int = 0


def main() -> None:
    parser = HfArgumentParser((OPDTAMScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    if not _cli_arg_was_provided("--learning_rate", "--learning-rate"):
        training_args.learning_rate = DEFAULT_LEARNING_RATE

    if not script_args.dataset_name:
        raise ValueError("--dataset_name is required (a HuggingFace dataset id).")
    if script_args.teacher_source != "local_hf":
        raise ValueError(
            "OPD TAM training requires teacher_source='local_hf' "
            "(the TAM term needs the teacher's last-layer hidden states)."
        )
    if not script_args.teacher_model_name_or_path:
        raise ValueError("--teacher_model_name_or_path is required for local_hf.")

    # HfArgumentParser parses bools loosely; normalize the TAM toggles.
    script_args.tam_use_eci = _as_bool(script_args.tam_use_eci)
    script_args.tam_detach_lm_head = _as_bool(script_args.tam_detach_lm_head)
    script_args.tam_blur = _as_bool(script_args.tam_blur)
    script_args.tam_gate = _as_bool(script_args.tam_gate)

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
            os.path.basename(os.path.normpath(training_args.output_dir)) or "opd_tam"
        )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("\n" + "=" * 80)
        print("OPD + TAM VISUAL-EVIDENCE ALIGNMENT RUN CONFIGURATION")
        print("=" * 80)
        print(f"Student model: {script_args.model_name_or_path}")
        print(f"Teacher model: {script_args.teacher_model_name_or_path}")
        print(f"Finetuning mode: {script_args.finetuning_mode}")
        print(f"Freeze vision tower: {script_args.freeze_vision_tower}")
        print(
            "OPD: "
            f"loss_mode={script_args.opd_loss_mode}, "
            f"kl_direction={script_args.opd_kl_direction}, "
            f"lambda_opd={script_args.lambda_opd}"
        )
        print(
            "TAM: "
            f"lambda_tam={script_args.lambda_tam}, "
            f"span={script_args.tam_align_span}, "
            f"divergence={script_args.tam_divergence}, "
            f"eci={script_args.tam_use_eci}, "
            f"denoise={script_args.tam_denoise}(k={script_args.tam_blur_kernel},"
            f"sigma={script_args.tam_blur_sigma}), "
            f"gate={script_args.tam_gate}(h0={script_args.tam_gate_h0},tau={script_args.tam_gate_tau}), "
            f"max_tokens={script_args.tam_max_tokens}"
        )
        print(f"Output directory: {training_args.output_dir}")
        print(f"Dataset name: {script_args.dataset_name}  (split={script_args.dataset_split})")
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
    # Qwen embeds a padded vocab (e.g. 151936) wider than the tokenizer's real
    # string vocab, so on-policy sampling can emit an id in the padded region.
    # The slow tokenizer maps such an id to None and then dies in "".join(tokens)
    # while decoding the rollout — one bad token kills the whole run. Coalesce
    # None -> "" so an unmapped id is dropped instead of crashing.
    if hasattr(tokenizer, "_convert_id_to_token"):
        _orig_id_to_token = tokenizer._convert_id_to_token

        def _safe_id_to_token(index, _orig=_orig_id_to_token):
            token = _orig(index)
            return "" if token is None else token

        tokenizer._convert_id_to_token = _safe_id_to_token

    # Cap image resolution to bound the hidden-state retention of output_hidden_states.
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
        dataset = dataset.select(range(min(script_args.max_train_samples, len(dataset))))
    if script_args.filter_tiny_images:
        pre = len(dataset)
        dataset = dataset_utils.filter_tiny_image_samples(
            dataset, min_image_size=script_args.min_image_size
        )
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"Dataset filtering removed {pre - len(dataset)}/{pre} samples.")
    if os.environ.get("LOCAL_RANK", "0") == "0":
        cap = (
            f"  (capped by max_train_samples={script_args.max_train_samples})"
            if script_args.max_train_samples is not None
            else ""
        )
        print(
            f"[OPD-TAM] dataset={script_args.dataset_name} split={script_args.dataset_split} "
            f"-> {len(dataset)} training examples{cap}",
            flush=True,
        )

    data_collator = OPDDataCollator(
        processor=processor,
        max_prompt_length=script_args.max_prompt_length,
        answer_field=script_args.answer_field,
        opd_prompt_suffix=script_args.opd_prompt_suffix,
    )

    # --- Trainer ----------------------------------------------------------------
    trainer = TAMTrainer(
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
        # TAM knobs
        lambda_tam=script_args.lambda_tam,
        tam_align_span=script_args.tam_align_span,
        tam_use_eci=script_args.tam_use_eci,
        tam_detach_lm_head=script_args.tam_detach_lm_head,
        tam_divergence=script_args.tam_divergence,
        tam_blur=script_args.tam_blur,
        tam_denoise=script_args.tam_denoise,
        tam_blur_kernel=script_args.tam_blur_kernel,
        tam_blur_sigma=script_args.tam_blur_sigma,
        tam_gate=script_args.tam_gate,
        tam_gate_temp=script_args.tam_gate_temp,
        tam_gate_h0=script_args.tam_gate_h0,
        tam_gate_tau=script_args.tam_gate_tau,
        tam_mass_threshold=script_args.tam_mass_threshold,
        tam_max_tokens=script_args.tam_max_tokens,
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
                    "opd_method": "opd_tam",
                    "opd_student_model": script_args.model_name_or_path,
                    "opd_teacher_model": script_args.teacher_model_name_or_path,
                    "opd_dataset_name": script_args.dataset_name,
                    "opd_lambda_opd": script_args.lambda_opd,
                    "tam_lambda": script_args.lambda_tam,
                    "tam_align_span": script_args.tam_align_span,
                    "tam_divergence": script_args.tam_divergence,
                    "tam_use_eci": script_args.tam_use_eci,
                    "tam_blur": script_args.tam_blur,
                    "tam_denoise": script_args.tam_denoise,
                    "tam_gate": script_args.tam_gate,
                    "tam_gate_h0": script_args.tam_gate_h0,
                    "tam_gate_tau": script_args.tam_gate_tau,
                    "tam_max_tokens": script_args.tam_max_tokens,
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
            f"[OPD-TAM] num_processes(world_size)={num_proc}  "
            f"per_device_bs={training_args.per_device_train_batch_size}  "
            f"grad_accum={training_args.gradient_accumulation_steps}  "
            f"-> effective_batch={eff}  lambda_tam={script_args.lambda_tam}",
            flush=True,
        )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
