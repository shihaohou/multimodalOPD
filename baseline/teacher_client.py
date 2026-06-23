"""Client for the vLLM-server OPD teacher (top-k logprob scoring).

The trainer sends, per sample, the exact ``prompt_token_ids + completion_token_ids``
(already tokenized by the student processor — teacher shares the vocab) plus the
image, and the server returns the teacher's **top-k** token ids + log-probs at each
completion position (via vLLM ``prompt_logprobs``). This removes the per-GPU frozen
teacher replica and lets the teacher be much larger than the student.

EXPERIMENTAL: the multimodal ``prompt_token_ids`` + ``prompt_logprobs`` path needs
GPU validation; ``teacher_source=local_hf`` remains the default elsewhere.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.request
from typing import Any

import torch
from PIL import Image


class VLLMServerTeacher:
    def __init__(
        self,
        server_url: str,
        *,
        top_k: int = 32,
        timeout: float = 120.0,
        retries: int = 2,
    ) -> None:
        self.endpoint = server_url.rstrip("/") + "/score_topk"
        self.top_k = int(top_k)
        self.timeout = float(timeout)
        self.retries = int(retries)

    def score_topk(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_attention_mask: torch.Tensor,
        images: list[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (topk_ids [B,C,k], topk_logprobs [B,C,k]) on completion_ids.device.

        Unused top-k slots and padded completion positions use id ``-1`` so the
        loss can mask them.
        """
        device = completion_ids.device
        batch_size, comp_len = completion_ids.shape
        prompt_mask = prompt_attention_mask.to(dtype=torch.bool)
        comp_mask = completion_attention_mask.to(dtype=torch.bool)

        items = []
        comp_lengths = []
        for row in range(batch_size):
            prompt_ids = prompt_input_ids[row][prompt_mask[row]].tolist()
            length = int(comp_mask[row].sum().item())
            comp_lengths.append(length)
            completion = completion_ids[row][:length].tolist()
            items.append(
                {
                    "prompt_token_ids": [int(x) for x in prompt_ids],
                    "completion_token_ids": [int(x) for x in completion],
                    "image_b64": _encode_image(images[row]) if row < len(images) else None,
                }
            )

        response = self._post({"top_k": self.top_k, "items": items})

        topk_ids = torch.full(
            (batch_size, comp_len, self.top_k), -1, dtype=torch.long, device=device
        )
        topk_logprobs = torch.zeros(
            (batch_size, comp_len, self.top_k), dtype=torch.float32, device=device
        )
        for row, item in enumerate(response["items"]):
            ids = item.get("topk_ids") or []
            logprobs = item.get("topk_logprobs") or []
            for pos in range(min(comp_lengths[row], len(ids), comp_len)):
                row_ids = ids[pos][: self.top_k]
                row_lps = logprobs[pos][: self.top_k]
                width = min(len(row_ids), self.top_k)
                if width:
                    topk_ids[row, pos, :width] = torch.tensor(
                        row_ids[:width], dtype=torch.long, device=device
                    )
                    topk_logprobs[row, pos, :width] = torch.tensor(
                        row_lps[:width], dtype=torch.float32, device=device
                    )
        return topk_ids, topk_logprobs

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                request = urllib.request.Request(
                    self.endpoint,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                    return json.loads(handle.read().decode("utf-8"))
            except Exception as exc:  # noqa: BLE001 - retry network/server errors
                last_error = exc
        raise RuntimeError(
            f"vLLM teacher server request to {self.endpoint} failed: {last_error}"
        )


def _encode_image(image: Any) -> str | None:
    if image is None:
        return None
    if isinstance(image, (list, tuple)):
        image = image[0] if image else None
        if image is None:
            return None
    if not isinstance(image, Image.Image):
        return None
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
