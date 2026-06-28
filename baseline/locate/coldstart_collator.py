"""SFT collation for the Locate-Once cold-start.

Standard supervised fine-tuning: build the **locate** prompt (system =
``LOCATE_SYSTEM_PROMPT`` + user(image + question)) and supervise the cold-start
``target`` completion (``<think><box>[GT]</box> reasoning </think> \\boxed{answer}``).
Prompt tokens (incl. image placeholders) are masked to ``-100``; the target + EOS are
supervised. Right-padded (training), unlike the rollout collators' left padding.

The model's own forward computes the cross-entropy from ``labels`` — no custom loss
needed, so :mod:`baseline.locate.coldstart_sft` just uses a vanilla HF ``Trainer``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from baseline.locate.prompts import LOCATE_SYSTEM_PROMPT
from baseline.opd_data_collator import _safe_rgb_image, build_opd_messages

# Loss-ignore index for masked (prompt + pad) positions.
IGNORE_INDEX = -100


def assemble_sft_row(
    prompt_ids: list[int], target_ids: list[int], eos_id: int
) -> tuple[list[int], list[int]]:
    """``(input_ids, labels)`` for one SFT row (pure; CPU-testable).

    ``input_ids = prompt + target + EOS``; ``labels`` masks the whole prompt with
    ``IGNORE_INDEX`` and supervises ``target + EOS`` (so the model also learns to stop).
    """
    target = list(target_ids) + [int(eos_id)]
    input_ids = list(prompt_ids) + target
    labels = [IGNORE_INDEX] * len(prompt_ids) + target
    return input_ids, labels


@dataclass
class ColdStartSFTCollator:
    processor: Any
    locate_system_prompt: str = LOCATE_SYSTEM_PROMPT
    max_prompt_length: int = 8192
    max_target_length: int = 1024
    tokenizer: Any = field(default=None, init=False)
    pad_id: int = field(default=0, init=False)
    eos_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        eos = getattr(self.tokenizer, "eos_token_id", None)
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if pad is None:
            pad = eos
        if eos is None:
            eos = pad
        if pad is None or eos is None:
            raise ValueError("Tokenizer must define a pad or eos token for SFT collation.")
        self.pad_id = int(pad)
        self.eos_id = int(eos)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        attn_rows: list[torch.Tensor] = []
        pixel_values_list: list[torch.Tensor] = []
        grid_list: list[torch.Tensor] = []

        for feature in features:
            image = _safe_rgb_image(feature.get("image", feature.get("images")))
            problem = str(feature["problem"])
            target = str(feature["target"])
            messages = build_opd_messages(
                problem, image, system_prompt=self.locate_system_prompt, suffix=""
            )
            enc = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_length,
            )
            prompt_ids = enc["input_ids"][0].tolist()
            target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
            if self.max_target_length and len(target_ids) > self.max_target_length - 1:
                target_ids = target_ids[: self.max_target_length - 1]
            input_ids, labels = assemble_sft_row(prompt_ids, target_ids, self.eos_id)

            input_rows.append(torch.tensor(input_ids, dtype=torch.long))
            label_rows.append(torch.tensor(labels, dtype=torch.long))
            attn_rows.append(torch.ones(len(input_ids), dtype=torch.long))
            if "pixel_values" in enc:
                pixel_values_list.append(enc["pixel_values"])
            if "image_grid_thw" in enc:
                grid_list.append(enc["image_grid_thw"])

        max_len = max(row.shape[0] for row in input_rows)

        def _pad(rows: list[torch.Tensor], value: int) -> torch.Tensor:
            return torch.stack(
                [
                    torch.cat([row, row.new_full((max_len - row.shape[0],), value)])
                    for row in rows
                ]
            )

        batch: dict[str, torch.Tensor] = {
            "input_ids": _pad(input_rows, self.pad_id),
            "attention_mask": _pad(attn_rows, 0),
            "labels": _pad(label_rows, IGNORE_INDEX),
        }
        # Qwen packs image patches as [sum_patches, hidden] with image_grid_thw [n,3];
        # concatenate across the batch (matches the processor's own batched output).
        if pixel_values_list:
            batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if grid_list:
            batch["image_grid_thw"] = torch.cat(grid_list, dim=0)
        return batch
