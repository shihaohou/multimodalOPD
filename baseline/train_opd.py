"""Entry point for vanilla multimodal On-Policy Distillation (OPD).

Standalone counterpart to ``vigos/train_vigos.py``. It trains a LoRA student
against a separate, frozen, stronger same-family VLM teacher using on-policy
reverse-KL distillation (see :class:`baseline.opd_trainer.OPDTrainer`). ViGOS /
OPSD code paths are left completely untouched; this only reuses the ``vigos``
framework as a library.

Example
-------
    DATASET_NAME=LMMs-Lab-Turtle/Vision-SR1-47K \\
    TEACHER_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \\
    bash scripts/train_opd.sh
"""

from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoProcessor,
    HfArgumentParser,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:  # transformers<=4.57.1 may not yet ship Qwen3.5.
    Qwen3_5ForConditionalGeneration = None

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:
    AutoModelForMultimodalLM = None

import vigos.dataset_utils as dataset_utils
from baseline.opd_data_collator import OPDDataCollator, resolve_opd_system_prompt
from baseline.opd_trainer import OPDTrainer
from vigos.train_vigos import (
    DEFAULT_LEARNING_RATE,
    _cli_arg_was_provided,
    _dtype,
    _model_class_for_checkpoint,
    _reporting_to_wandb,
)

from baseline.opd_dataset import load_opd_dataset


def _opd_model_class_for_checkpoint(
    model_name_or_path: str,
    trust_remote_code: bool = True,
):
    """Resolve the VL model class, tolerating a ``*_text`` top-level ``model_type``.

    ``vigos._model_class_for_checkpoint`` only accepts the full VL model_type
    (``qwen2_5_vl`` / ``qwen3_vl``). Some checkpoints report the **text sub-config's**
    type at the top level instead — e.g. ``Saliency-R1-7B`` (and ms-swift / re-saved
    merges) come up as ``qwen2_5_vl_text`` — even though the checkpoint is still a
    full VL model (vision tower included). Strip a trailing ``_text`` and resolve by
    family; the concrete class's ``from_pretrained`` then parses the *full*
    config.json (vision_config and all) via its own ``config_class``, so the
    mislabeled ``model_type`` string is cosmetic. Unknown types defer to the strict
    vigos resolver so its error message/contract is preserved.

    (Lives in the OPD layer so ``vigos/`` stays untouched.)
    """
    try:
        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        model_type = getattr(config, "model_type", "") or ""
    except Exception:
        config_path = Path(model_name_or_path) / "config.json"
        if not config_path.exists():
            raise
        with config_path.open("r", encoding="utf-8") as handle:
            model_type = str(json.load(handle).get("model_type", "") or "")
    base = model_type[: -len("_text")] if model_type.endswith("_text") else model_type
    if base == "qwen2_5_vl":
        return Qwen2_5_VLForConditionalGeneration, model_type
    if base == "qwen3_vl":
        return Qwen3VLForConditionalGeneration, model_type
    if base == "qwen3_5":
        qwen35_class = Qwen3_5ForConditionalGeneration or AutoModelForMultimodalLM
        if qwen35_class is None:
            raise ValueError(
                "Qwen3.5 checkpoints use model_type='qwen3_5' (or a top-level "
                "'qwen3_5_text' in some re-saves) and require a Transformers build "
                "that exports Qwen3_5ForConditionalGeneration or "
                "AutoModelForMultimodalLM. Install a newer Transformers/source build; "
                "do not load Qwen3.5 with the Qwen3-VL model class because the text "
                "architecture differs."
            )
        return qwen35_class, model_type
    return _model_class_for_checkpoint(model_name_or_path, trust_remote_code)

# ViRL39K-aware loader (local parquet -> problem/image/answer); falls through to
# vigos.dataset_utils.load_vigos_dataset for HuggingFace ids / canonical datasets.
_load_dataset = load_opd_dataset
_filter_tiny_image_samples = dataset_utils.filter_tiny_image_samples


@dataclass
class OPDScriptArguments:
    model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    teacher_model_name_or_path: str | None = None
    teacher_torch_dtype: str = "bfloat16"
    teacher_attn_implementation: str = "flash_attention_2"
    # Teacher source: "local_hf" = frozen replica per GPU (full-vocab KL options);
    # "vllm_server" = separate vLLM server returning top-k logprobs (forward top-k
    # KL only; no per-GPU replica, supports much larger teachers).
    teacher_source: str = "local_hf"
    teacher_server_url: str = "http://127.0.0.1:8200"
    teacher_client_timeout: float = 120.0
    teacher_client_retries: int = 2
    dataset_name: str | None = None
    dataset_split: str = "train"
    max_train_samples: Optional[int] = None
    filter_tiny_images: bool = False
    min_image_size: int = 3
    max_prompt_length: int = 32768
    max_completion_length: int = 4096
    attn_implementation: str = "flash_attention_2"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    finetuning_mode: str = "full"  # "full" (default, like Vision-OPD) | "lora"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    # Comma-separated module list (LLM only by default), or "all-linear" to also
    # cover the vision tower + merger (ViT+LLM LoRA).
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    answer_field: str = "answer"
    # Format instruction now lives in the unified system prompt; user = raw question.
    opd_prompt_suffix: str = ""
    # System-prompt style for the rollout (the frozen teacher is scored on the same
    # prompt, so this also controls it): "think" (default, <think></think> tags) |
    # "freecot" (OPD-main free-text CoT, no tags) | "reason" (<reason></reason> tags) |
    # "none", or a raw system-prompt string. See opd_data_collator.OPD_SYSTEM_PROMPTS.
    opd_system_prompt: str = "think"
    generation_temperature: float = 1.1
    generation_top_p: float = 0.95
    generation_top_k: int = 20
    distill_temperature: float = 1.0
    lambda_opd: float = 1.0
    # Distillation divergence. Default = top-k reverse KL (top-100): the OPD-ecosystem
    # standard (verl/Uni-OPD/thunlp-OPD), ~99% of the mass, and it avoids the
    # full-vocab exp/diff that OOMs at micro-batch 8 on long completions. Set
    # full_kl for exact full-vocab KL (canonical but heavier).
    opd_loss_mode: str = "topk_kl"  # "topk_kl" | "full_kl"
    opd_kl_direction: str = "reverse"  # "reverse" | "forward" | "jsd"
    opd_top_k: int = 100
    # Freeze the Qwen-VL vision tower under full FT. Off by default: at a real
    # multi-GPU effective batch (e.g. 32) the ViT gradient spikes average out and
    # full FT incl. ViT is stable (matches Vision-OPD). Turn on only as a fallback
    # for small-batch / single-GPU runs where the ViT bf16 grad can overflow.
    # Only applies to finetuning_mode=full.
    freeze_vision_tower: bool = False
    token_loss_clip: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_p: float = 0.0
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.3
    vllm_tensor_parallel_size: int = 1
    vllm_sync_frequency: int = 1
    vllm_max_model_len: Optional[int] = None
    vllm_max_num_seqs: Optional[int] = None
    vllm_disable_custom_all_reduce: bool = False
    vllm_server_base_url: Optional[str] = None
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 8000
    vllm_server_timeout: float = 300.0
    vllm_server_group_port: int = 51216
    vllm_server_request_batch_size: Optional[int] = None
    completion_log_steps: int = 0
    completion_log_max_samples: int = 16
    run_config: str | None = field(
        default=None,
        metadata={"help": "Run config label; appended to output_dir and WandB run name."},
    )
    run_name_suffix: str | None = field(
        default=None,
        metadata={"help": "Optional suffix appended to the Trainer run name."},
    )


class _OPDWandBConfigCallback(TrainerCallback):
    """Log static run config to WandB once at train start (best effort)."""

    def __init__(self, config: dict[str, object]) -> None:
        self._config = config
        self._logged = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self._logged:
            return control
        self._logged = True
        try:
            import wandb

            if wandb.run is not None:
                wandb.config.update(self._config, allow_val_change=True)
        except Exception:
            pass
        return control


def main() -> None:
    parser = HfArgumentParser((OPDScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    if not _cli_arg_was_provided("--learning_rate", "--learning-rate"):
        training_args.learning_rate = DEFAULT_LEARNING_RATE

    if not script_args.dataset_name:
        raise ValueError("--dataset_name is required (a HuggingFace dataset id).")
    if script_args.teacher_source == "local_hf" and not script_args.teacher_model_name_or_path:
        raise ValueError(
            "teacher_source='local_hf' requires --teacher_model_name_or_path "
            "(a stronger same-family VLM checkpoint; base or RL-tuned)."
        )
    if script_args.teacher_source == "vllm_server" and not script_args.teacher_server_url:
        raise ValueError("teacher_source='vllm_server' requires --teacher_server_url.")

    if script_args.run_config:
        lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        effective_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * world_size
        )
        training_args.run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not Path(training_args.output_dir).name == script_args.run_config:
            training_args.output_dir = str(
                Path(training_args.output_dir) / script_args.run_config
            )
    elif script_args.run_name_suffix:
        base_name = training_args.run_name or "opd"
        training_args.run_name = f"{base_name}_{script_args.run_name_suffix}"
    elif not training_args.run_name or training_args.run_name == training_args.output_dir:
        training_args.run_name = (
            os.path.basename(os.path.normpath(training_args.output_dir)) or "opd"
        )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("\n" + "=" * 80)
        print("OPD RUN CONFIGURATION")
        print("=" * 80)
        print(f"Student model: {script_args.model_name_or_path}")
        print(f"Teacher source: {script_args.teacher_source}")
        if script_args.teacher_source == "vllm_server":
            print(f"Teacher server: {script_args.teacher_server_url}")
        else:
            print(f"Teacher model: {script_args.teacher_model_name_or_path}")
        print(f"Finetuning mode: {script_args.finetuning_mode}")
        print(f"WandB/Trainer run name: {training_args.run_name}")
        print(f"Output directory: {training_args.output_dir}")
        print(f"Dataset name: {script_args.dataset_name}")
        print(f"Answer/reference field: {script_args.answer_field}")
        print(f"OPD prompt suffix: {script_args.opd_prompt_suffix!r}")
        print(
            f"OPD system prompt: style={script_args.opd_system_prompt!r} -> "
            f"{resolve_opd_system_prompt(script_args.opd_system_prompt)!r}"
        )
        print(
            "Rollout sampling: "
            f"temperature={script_args.generation_temperature}, "
            f"top_p={script_args.generation_top_p}, "
            f"top_k={script_args.generation_top_k}"
        )
        print(
            "Distillation: "
            f"loss_mode={script_args.opd_loss_mode}, "
            f"kl_direction={script_args.opd_kl_direction}, "
            f"top_k={script_args.opd_top_k}, "
            f"lambda_opd={script_args.lambda_opd}, "
            f"distill_temperature={script_args.distill_temperature}, "
            f"token_loss_clip={script_args.token_loss_clip}"
        )
        print(
            "vLLM: "
            f"use_vllm={script_args.use_vllm}, mode={script_args.vllm_mode}, "
            f"tp={script_args.vllm_tensor_parallel_size}, "
            f"gpu_memory_utilization={script_args.vllm_gpu_memory_utilization}"
        )
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

    # --- Student (trainable LoRA) -------------------------------------------------
    model_kwargs = {
        "trust_remote_code": script_args.trust_remote_code,
        "attn_implementation": script_args.attn_implementation,
        "dtype": _dtype(script_args.torch_dtype),
    }
    model_class, model_type = _opd_model_class_for_checkpoint(
        script_args.model_name_or_path,
        trust_remote_code=script_args.trust_remote_code,
    )
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"Resolved student model_type={model_type} to {model_class.__name__}.")
    model = model_class.from_pretrained(script_args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False if training_args.gradient_checkpointing else True
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if script_args.finetuning_mode == "full":
        # Full-parameter training (default; matches Vision-OPD). save_model writes
        # the full checkpoint, so no LoRA merge step is needed before eval.
        if script_args.freeze_vision_tower:
            # The Qwen-VL vision tower (ViT conv patch_embed) is numerically
            # unstable under bf16 full fine-tuning: its gradient overflows and
            # poisons the Adam state -> NaN weights within a few steps (the OPD
            # NaN probe pinned `visual.patch_embed.proj.weight` as the first
            # tensor to go non-finite). Distillation doesn't need to retrain the
            # ViT, so freeze it (standard for VLM post-training / RL).
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
        # "all-linear" is a PEFT sentinel: attach LoRA to every nn.Linear except
        # the output head — i.e. the vision tower + multimodal merger + LLM all
        # get adapters (the only robust way to cover the ViT without hardcoding
        # its module names). Anything else is an explicit comma-separated list.
        raw_targets = script_args.lora_target_modules.strip()
        if raw_targets == "all-linear":
            target_modules = "all-linear"
        else:
            target_modules = [
                module.strip()
                for module in raw_targets.split(",")
                if module.strip()
            ]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        raise ValueError(
            f"Unknown finetuning_mode {script_args.finetuning_mode!r}; "
            "expected 'full' or 'lora'."
        )

    # --- Teacher -----------------------------------------------------------------
    teacher_model = None
    teacher_client = None
    if script_args.teacher_source == "local_hf":
        teacher_class, teacher_type = _opd_model_class_for_checkpoint(
            script_args.teacher_model_name_or_path,
            trust_remote_code=script_args.trust_remote_code,
        )
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(
                f"Loading OPD teacher {script_args.teacher_model_name_or_path} "
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
    else:  # vllm_server: query a separate teacher server for top-k logprobs.
        from baseline.teacher_client import VLLMServerTeacher

        teacher_client = VLLMServerTeacher(
            script_args.teacher_server_url,
            top_k=script_args.opd_top_k,
            timeout=script_args.teacher_client_timeout,
            retries=script_args.teacher_client_retries,
        )
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"Using vLLM teacher server at {script_args.teacher_server_url}.")

    # --- Data ---------------------------------------------------------------------
    dataset = _load_dataset(script_args.dataset_name, script_args.dataset_split)
    if script_args.max_train_samples is not None:
        dataset = dataset.select(range(min(script_args.max_train_samples, len(dataset))))
    if script_args.filter_tiny_images:
        pre = len(dataset)
        dataset = _filter_tiny_image_samples(
            dataset, min_image_size=script_args.min_image_size
        )
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"Dataset filtering removed {pre - len(dataset)}/{pre} samples.")

    data_collator = OPDDataCollator(
        processor=processor,
        max_prompt_length=script_args.max_prompt_length,
        answer_field=script_args.answer_field,
        opd_prompt_suffix=script_args.opd_prompt_suffix,
        system_prompt=resolve_opd_system_prompt(script_args.opd_system_prompt),
    )

    # --- Trainer ------------------------------------------------------------------
    trainer = OPDTrainer(
        model=model,
        args=training_args,
        model_name_or_path=script_args.model_name_or_path,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=processor,
        processor=processor,
        teacher_model=teacher_model,
        teacher_source=script_args.teacher_source,
        teacher_client=teacher_client,
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
        vllm_server_base_url=script_args.vllm_server_base_url,
        vllm_server_host=script_args.vllm_server_host,
        vllm_server_port=script_args.vllm_server_port,
        vllm_server_timeout=script_args.vllm_server_timeout,
        vllm_server_group_port=script_args.vllm_server_group_port,
        vllm_server_request_batch_size=script_args.vllm_server_request_batch_size,
        completion_log_steps=script_args.completion_log_steps,
        completion_log_max_samples=script_args.completion_log_max_samples,
    )

    if _reporting_to_wandb(training_args):
        trainer.add_callback(
            _OPDWandBConfigCallback(
                {
                    "opd_method": "opd",
                    "opd_finetuning_mode": script_args.finetuning_mode,
                    "opd_student_model": script_args.model_name_or_path,
                    "opd_teacher_source": script_args.teacher_source,
                    "opd_teacher_model": script_args.teacher_model_name_or_path,
                    "opd_teacher_server_url": (
                        script_args.teacher_server_url
                        if script_args.teacher_source == "vllm_server"
                        else None
                    ),
                    "opd_dataset_name": script_args.dataset_name,
                    "opd_train_dataset_size": len(dataset),
                    "opd_lambda_opd": script_args.lambda_opd,
                    "opd_distill_temperature": script_args.distill_temperature,
                    "opd_token_loss_clip": (
                        script_args.token_loss_clip
                        if script_args.token_loss_clip > 0
                        else None
                    ),
                    "opd_loss_mode": script_args.opd_loss_mode,
                    "opd_kl_direction": script_args.opd_kl_direction,
                    "opd_top_k": script_args.opd_top_k,
                    "opd_prompt_suffix": script_args.opd_prompt_suffix,
                    "opd_system_prompt": script_args.opd_system_prompt,
                    "opd_max_prompt_length": script_args.max_prompt_length,
                    "opd_max_completion_length": script_args.max_completion_length,
                }
            )
        )

    # Authoritative GPU/world-size report (from the live accelerator, not the env
    # var) so every run self-documents whether it is actually multi-GPU.
    if trainer.accelerator.is_main_process:
        num_proc = trainer.accelerator.num_processes
        eff_batch = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * num_proc
        )
        print(
            f"[OPD] num_processes(world_size)={num_proc}  "
            f"per_device_bs={training_args.per_device_train_batch_size}  "
            f"grad_accum={training_args.gradient_accumulation_steps}  "
            f"-> effective_batch={eff_batch}",
            flush=True,
        )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
