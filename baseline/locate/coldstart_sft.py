"""Cold-start SFT for Locate-Once Grounding.

Standard supervised fine-tuning that teaches the student to emit the locate-once
format (``<think><box>[x1,y1,x2,y2]</box> reasoning </think> \\boxed{answer}`` with
[0,1] coords) on the traces built by :mod:`baseline.locate.coldstart_build`. A vanilla
HF ``Trainer`` — the model's own forward computes the cross-entropy from ``labels``
(prompt masked), so there is no custom loss / teacher / rollout here. The resulting
checkpoint becomes ``MODEL_NAME_OR_PATH`` for the RL+OPD locate run
(``train_opd_locate.py``), which then has ``box_coverage`` > 0 so the RL term fires.

Run (8 GPU, DeepSpeed):
    uv run accelerate launch --config_file configs/accelerate_zero2_gpu_8.yaml \\
        baseline/locate/coldstart_sft.py \\
        --model_name_or_path $M/Qwen3-VL-2B-Instruct \\
        --dataset_dir runs/coldstart_locate_traces \\
        --output_dir runs/qwen3_2b_locate_coldstart --bf16 --gradient_checkpointing
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets import load_from_disk
from transformers import AutoProcessor, HfArgumentParser, Trainer, TrainingArguments, set_seed

from baseline.locate.coldstart_collator import ColdStartSFTCollator
from baseline.locate.prompts import LOCATE_SYSTEM_PROMPT
from baseline.train_opd import _opd_model_class_for_checkpoint
from vigos.train_vigos import _dtype


@dataclass
class ColdStartSFTArguments:
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    dataset_dir: str | None = None  # save_to_disk dir from coldstart_build
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = True
    finetuning_mode: str = "full"  # "full" | "lora"
    freeze_vision_tower: bool = False
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    locate_system_prompt: str = LOCATE_SYSTEM_PROMPT
    max_prompt_length: int = 8192
    max_target_length: int = 1024


def main() -> None:
    parser = HfArgumentParser((ColdStartSFTArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False  # keep image/problem/target for the collator
    if not script_args.dataset_dir:
        raise ValueError("--dataset_dir is required (the coldstart_build save_to_disk dir).")
    if "<box>" not in script_args.locate_system_prompt:
        raise ValueError("--locate_system_prompt must request a <box> (it is what SFT teaches).")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        print("\n" + "=" * 80)
        print("LOCATE-ONCE COLD-START SFT")
        print("=" * 80)
        print(f"Student model: {script_args.model_name_or_path}")
        print(f"Cold-start traces: {script_args.dataset_dir}")
        print(f"Finetuning mode: {script_args.finetuning_mode}")
        print(f"Output: {training_args.output_dir}")
        print("=" * 80 + "\n")

    set_seed(training_args.seed)

    processor = AutoProcessor.from_pretrained(
        script_args.model_name_or_path, trust_remote_code=script_args.trust_remote_code, use_fast=False
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # training (the collator also right-pads)

    model_class, model_type = _opd_model_class_for_checkpoint(
        script_args.model_name_or_path, trust_remote_code=script_args.trust_remote_code
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
            for name, param in model.named_parameters():
                if "visual." in name:
                    param.requires_grad_(False)
    elif script_args.finetuning_mode == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        raw = script_args.lora_target_modules.strip()
        targets = "all-linear" if raw == "all-linear" else [m.strip() for m in raw.split(",") if m.strip()]
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=script_args.lora_r,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                target_modules=targets,
                bias="none",
            ),
        )
        model.print_trainable_parameters()
    else:
        raise ValueError(f"Unknown finetuning_mode {script_args.finetuning_mode!r}.")

    dataset = load_from_disk(script_args.dataset_dir)
    data_collator = ColdStartSFTCollator(
        processor=processor,
        locate_system_prompt=script_args.locate_system_prompt,
        max_prompt_length=script_args.max_prompt_length,
        max_target_length=script_args.max_target_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=processor,
    )
    if trainer.accelerator.is_main_process:
        print(f"[coldstart-sft] {len(dataset)} traces, output -> {training_args.output_dir}", flush=True)

    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
