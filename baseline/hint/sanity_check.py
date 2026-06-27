"""Sanity check for Grounding-Hint Distillation (GHD) data collation.

Two parts:

* ``check_hint_text`` — no model/processor needed. Verifies the hint string and
  message construction: coordinates are rendered correctly, the hint lands in the
  teacher's user turn, and an empty hint reproduces the student text.
* ``check_collator`` — needs the student processor (``--model``). Builds a tiny
  2-row batch (one row with a box, one without), runs ``OPDHintDataCollator``, and
  asserts the privileged ``teacher_prompt_*`` is built correctly: the hint text is
  present only for the boxed row, the teacher prompt is strictly longer there, the
  two prompts encode the *same* image (identical ``image_grid_thw``), and the
  ``has_hint`` flags match.

Run (text-only, anywhere):
    uv run python -m baseline.hint.sanity_check
Run (full, on a box with the model):
    uv run python -m baseline.hint.sanity_check --model /path/to/Qwen3-VL-2B-Instruct
"""

from __future__ import annotations

import argparse

from PIL import Image

from baseline.hint.opd_hint_collator import (
    HINT_TEMPLATE,
    build_hint_teacher_messages,
    crop_to_bbox,
    format_bbox_hint,
)
from baseline.opd_data_collator import OPD_SYSTEM_PROMPT


def _user_text(messages: list[dict]) -> str:
    """Concatenate the text parts of the (single) user turn."""
    user = [m for m in messages if m["role"] == "user"][-1]
    return "\n".join(
        part["text"] for part in user["content"] if part.get("type") == "text"
    )


def check_hint_text() -> None:
    box = (0.12, 0.34, 0.55, 0.78)
    hint = format_bbox_hint(box, HINT_TEMPLATE, decimals=2)
    assert "[0.12, 0.34, 0.55, 0.78]" in hint, hint
    assert "bounding box" in hint, hint

    problem = "What color is the bird's beak?"
    teacher_msgs = build_hint_teacher_messages(
        problem, image=None, hint=hint, system_prompt=OPD_SYSTEM_PROMPT
    )
    t_text = _user_text(teacher_msgs)
    assert problem in t_text and hint in t_text, t_text
    # Question comes before the hint (read the question, then where to look).
    assert t_text.index(problem) < t_text.index("bounding box"), t_text

    # Empty hint => the teacher user text is exactly the student question.
    student_like = build_hint_teacher_messages(
        problem, image=None, hint="", system_prompt=OPD_SYSTEM_PROMPT
    )
    assert _user_text(student_like) == problem, _user_text(student_like)

    # 3-decimal rendering knob.
    assert "[0.120, 0.340, 0.550, 0.780]" in format_bbox_hint(box, decimals=3)
    print("[ghd-sanity] check_hint_text OK")


def check_crop_geometry() -> None:
    img = Image.new("RGB", (1000, 500), (127, 127, 127))  # W=1000, H=500
    # box [0.1, 0.2, 0.5, 0.8] -> px (100,100,500,400) -> 400x300 crop
    crop = crop_to_bbox(img, (0.1, 0.2, 0.5, 0.8))
    assert crop.size == (400, 300), crop.size
    # padding 0.25 expands each side by 0.25*box -> x:[0,600] y:[25,475] -> 600x450
    padded = crop_to_bbox(img, (0.1, 0.2, 0.5, 0.8), padding=0.25)
    assert padded.size == (600, 450), padded.size
    # degenerate (zero-width) box -> original image, not a crash
    assert crop_to_bbox(img, (0.5, 0.2, 0.5, 0.8)).size == (1000, 500)
    # clamp to image bounds (box partly outside)
    assert crop_to_bbox(img, (0.8, 0.8, 1.0, 1.0)).size == (200, 100)
    print("[ghd-sanity] check_crop_geometry OK (400x300 / padded 600x450 / clamped)")


def check_collator(model: str) -> None:
    import torch
    from transformers import AutoProcessor

    from baseline.hint.opd_hint_collator import OPDHintDataCollator

    processor = AutoProcessor.from_pretrained(
        model, trust_remote_code=True, use_fast=False
    )
    collator = OPDHintDataCollator(
        processor=processor,
        max_prompt_length=4096,
        answer_field="solution",
        system_prompt=OPD_SYSTEM_PROMPT,
        bbox_field="bbox",
    )
    img = Image.new("RGB", (224, 224), (127, 127, 127))
    features = [
        {
            "image": img,
            "problem": "What color is the bird's beak?",
            "solution": "Yes",
            "bbox": "[0.12, 0.34, 0.55, 0.78]",
        },
        {  # no box -> teacher prompt should equal the student prompt for this row
            "image": img,
            "problem": "Read the title at the top of the document.",
            "solution": "Annual Report",
            "bbox": "",
        },
    ]
    out = collator(features)

    for key in ("teacher_prompt_input_ids", "teacher_prompt_attention_mask"):
        assert key in out, f"missing {key} (collator did not build the teacher prompt)"
    assert torch.equal(out["has_hint"], torch.tensor([1, 0])), out["has_hint"]

    s_text0, t_text0 = out["student_prompt_texts"][0], out["teacher_prompt_texts"][0]
    assert "bounding box" in t_text0 and "[0.12, 0.34, 0.55, 0.78]" in t_text0, t_text0
    assert "bounding box" not in s_text0, "hint leaked into the STUDENT prompt!"
    assert "bounding box" not in out["teacher_prompt_texts"][1], "hint on a box-less row!"

    s_mask = out["student_prompt_attention_mask"]
    t_mask = out["teacher_prompt_attention_mask"]
    s_len0, t_len0 = int(s_mask[0].sum()), int(t_mask[0].sum())
    s_len1, t_len1 = int(s_mask[1].sum()), int(t_mask[1].sum())
    assert t_len0 > s_len0, f"hint added no tokens (student={s_len0} teacher={t_len0})"
    assert t_len1 == s_len1, f"box-less row differs (student={s_len1} teacher={t_len1})"

    # Same image fed to both forwards: identical patch grid (direction, not pixels).
    if "student_prompt_image_grid_thw" in out:
        assert torch.equal(
            out["student_prompt_image_grid_thw"], out["teacher_prompt_image_grid_thw"]
        ), "teacher/student image grids differ — the teacher got a different image!"

    print(
        f"[ghd-sanity] check_collator(hint) OK  "
        f"(row0 student={s_len0} -> teacher={t_len0} tokens, +{t_len0 - s_len0} hint; "
        f"row1 box-less student==teacher={t_len1})"
    )

    # --- crop mode: teacher sees the cropped evidence image, NO hint text -------
    crop_collator = OPDHintDataCollator(
        processor=processor,
        max_prompt_length=4096,
        answer_field="solution",
        system_prompt=OPD_SYSTEM_PROMPT,
        bbox_field="bbox",
        teacher_privilege_mode="crop",
    )
    cout = crop_collator(features)
    assert torch.equal(cout["has_hint"], torch.tensor([1, 0])), cout["has_hint"]
    assert "bounding box" not in cout["teacher_prompt_texts"][0], "crop mode added hint text!"
    if "student_prompt_image_grid_thw" in cout:
        s_grid = cout["student_prompt_image_grid_thw"]
        t_grid = cout["teacher_prompt_image_grid_thw"]
        # box-less row uses the full image on both sides -> identical grid.
        assert torch.equal(s_grid[1], t_grid[1]), "box-less row got a cropped teacher image!"
        print(
            f"[ghd-sanity] check_collator(crop) OK  (row0 cropped grid "
            f"student={s_grid[0].tolist()} -> teacher={t_grid[0].tolist()}; row1 full==full)"
        )
    else:
        print("[ghd-sanity] check_collator(crop) OK (no image_grid_thw to compare)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=None,
        help="Student model dir/id for the full collator check (needs the processor).",
    )
    args = ap.parse_args()
    check_hint_text()
    check_crop_geometry()
    if args.model:
        check_collator(args.model)
    else:
        print("[ghd-sanity] skipped check_collator (pass --model to run it)")


if __name__ == "__main__":
    main()
