"""Sanity check for Locate-Once Grounding (LOG).

Text-only (no model needed) — the RL math and prompt wiring:

* ``check_box_parsing``  — ``<box>`` extraction + normalization from a completion.
* ``check_iou``          — IoU of normalized boxes (overlap / disjoint / identical).
* ``check_advantage``    — GRPO group-normalized advantage (per-group, singleton=0).
* ``check_pg_gradient``  — the box PG gradient lands ONLY on box-coordinate positions,
  vanishes on non-box / no-box rows, and moves ``log pi`` in the advantage direction.
* ``check_prompts``      — the locate-once student prompt asks for a ``<box>``; an
  empty hint reproduces the question (teacher decoupling base case).

Full (needs the student processor, ``--model``):

* ``check_collator`` — builds a 2-prompt batch with ``group_size=3`` and asserts the
  group expansion (6 rows, correct ``group_ids`` / ``locate_gt_boxes``), that the
  STUDENT prompt carries the ``<box>`` instruction but NOT the hint, and the TEACHER
  prompt carries the hidden hint but NOT the locate instruction (the asymmetry).

Run (text-only, anywhere):
    uv run python -m baseline.locate.sanity_check
Run (full, on a box with the model):
    uv run python -m baseline.locate.sanity_check --model /path/to/Qwen3-VL-2B-Instruct
"""

from __future__ import annotations

import argparse

import torch

from baseline.locate.locate_rl import (
    extract_box_text,
    group_normalize_advantage,
    iou_norm,
    parse_student_box,
    sampled_token_logprobs,
)
from baseline.locate.prompts import LOCATE_SYSTEM_PROMPT


def check_box_parsing() -> None:
    completion = (
        "<think> <box>[0.12, 0.34, 0.55, 0.78]</box> the beak is in that region, "
        "it looks orange </think> \\boxed{orange}"
    )
    assert extract_box_text(completion) == "[0.12, 0.34, 0.55, 0.78]"
    box = parse_student_box(completion)
    assert box is not None and len(box) == 4, box
    assert all(abs(a - b) < 1e-6 for a, b in zip(box, (0.12, 0.34, 0.55, 0.78))), box

    # No box -> None (the rollout earns no localization credit / no RL handle).
    assert parse_student_box("<think> just reasoning </think> \\boxed{x}") is None
    # Coordinates without brackets still parse (ast.literal_eval path).
    assert parse_student_box("<box>0.1, 0.2, 0.5, 0.6</box>") == (0.1, 0.2, 0.5, 0.6)
    # FIRST box wins (a stray later mention can't hijack the reward).
    first = parse_student_box("<box>[0.1,0.1,0.2,0.2]</box> ... <box>[0.9,0.9,1,1]</box>")
    assert first == (0.1, 0.1, 0.2, 0.2), first
    # Degenerate (zero-area) -> None.
    assert parse_student_box("<box>[0.5, 0.5, 0.5, 0.9]</box>") is None
    print("[log-sanity] check_box_parsing OK")


def check_iou() -> None:
    a = (0.0, 0.0, 0.5, 0.5)
    assert abs(iou_norm(a, a) - 1.0) < 1e-9, "identical boxes -> IoU 1"
    # Disjoint -> 0.
    assert iou_norm((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0
    # Half overlap: a=[0,0,0.5,0.5] (area .25), b=[0.25,0,0.75,0.5] (area .25),
    # inter=[0.25,0,0.5,0.5] area .125 -> IoU = .125 / (.25+.25-.125) = 1/3.
    iou = iou_norm((0.0, 0.0, 0.5, 0.5), (0.25, 0.0, 0.75, 0.5))
    assert abs(iou - 1.0 / 3.0) < 1e-9, iou
    print("[log-sanity] check_iou OK")


def check_advantage() -> None:
    # Two groups of 2; group 0 rewards [0,1] -> centered [-.5,.5]; group 1 [1,1] -> [0,0].
    rewards = torch.tensor([0.0, 1.0, 1.0, 1.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    adv = group_normalize_advantage(rewards, group_ids, normalize_std=False)
    assert torch.allclose(adv, torch.tensor([-0.5, 0.5, 0.0, 0.0])), adv
    # std-normalized: group 0 population std = 0.5 -> [-1, 1]; constant group -> 0.
    adv_n = group_normalize_advantage(rewards, group_ids, normalize_std=True)
    assert torch.allclose(adv_n[:2], torch.tensor([-1.0, 1.0]), atol=1e-3), adv_n
    assert torch.allclose(adv_n[2:], torch.tensor([0.0, 0.0]), atol=1e-3), adv_n
    # Singleton group -> advantage 0 (no baseline).
    solo = group_normalize_advantage(torch.tensor([5.0]), torch.tensor([0]))
    assert torch.allclose(solo, torch.tensor([0.0])), solo
    # Advantage is detached (a weight, not a target).
    assert not adv.requires_grad
    print("[log-sanity] check_advantage OK")


def check_pg_gradient() -> None:
    """Gradient sanity for the box PG term (mirrors OPDLocateTrainer._box_rl_loss).

    Asserts the RL gradient lands ONLY on box-coordinate positions, vanishes on
    non-box positions and on rollouts with no box, and pushes log pi of the sampled
    token in the advantage direction (positive advantage -> increase its logit).
    """
    torch.manual_seed(0)
    batch, length, vocab = 2, 6, 40
    logits = torch.randn(batch, length, vocab, requires_grad=True)
    token_ids = torch.randint(0, vocab, (batch, length))
    # Row 0 has a box at cols 1..3; row 1 has none.
    box_mask = torch.zeros(batch, length, dtype=torch.bool)
    box_mask[0, 1:4] = True
    advantage = torch.tensor([2.0, -3.0])  # row0 positive, row1 negative (but no box)

    rows, cols = box_mask.nonzero(as_tuple=True)
    sel = logits[rows, cols].float()
    logp = sampled_token_logprobs(sel, token_ids[rows, cols])
    loss = -(advantage[rows] * logp).mean()
    loss.backward()
    grad = logits.grad

    assert torch.count_nonzero(grad[1]) == 0, "row with no box must get no RL gradient"
    assert torch.count_nonzero(grad[0, 0]) == 0, "non-box position (col 0) must get none"
    assert torch.count_nonzero(grad[0, 4:]) == 0, "non-box positions (cols 4+) must get none"
    assert grad[0, 1:4].abs().sum() > 0, "box positions must get gradient"
    # loss = -adv*logp with adv>0 => dloss/dlogit[sampled] < 0 (gradient step raises it).
    sampled = token_ids[0, 1]
    assert grad[0, 1, sampled] < 0, "positive advantage must increase sampled-token logit"
    # Full-vocab normalizer: gather - logsumexp matches manual log-softmax.
    manual = torch.log_softmax(logits[0, 1].detach().float(), dim=-1)[sampled]
    got = sampled_token_logprobs(logits[0, 1:2].detach().float(), token_ids[0, 1:2])[0]
    assert abs(float(got) - float(manual)) < 1e-5, (got, manual)
    print("[log-sanity] check_pg_gradient OK")


def check_prompts() -> None:
    assert "<box>" in LOCATE_SYSTEM_PROMPT, "student prompt must request a <box>"
    assert "\\boxed{}" in LOCATE_SYSTEM_PROMPT, "student must still answer in \\boxed{}"
    # The locate instruction must NOT mention a bounding-box HINT (that is teacher-only).
    assert "Hint:" not in LOCATE_SYSTEM_PROMPT
    print("[log-sanity] check_prompts OK")


def check_collator(model: str) -> None:
    from PIL import Image
    from transformers import AutoProcessor

    from baseline.opd_data_collator import OPD_SYSTEM_PROMPT
    from baseline.locate.opd_locate_collator import OPDLocateDataCollator

    processor = AutoProcessor.from_pretrained(
        model, trust_remote_code=True, use_fast=False
    )
    group_size = 3
    collator = OPDLocateDataCollator(
        processor=processor,
        max_prompt_length=4096,
        answer_field="solution",
        system_prompt=LOCATE_SYSTEM_PROMPT,       # student: locate-once
        teacher_system_prompt=OPD_SYSTEM_PROMPT,  # teacher: plain think (+ silent hint)
        bbox_field="bbox",
        group_size=group_size,
    )
    img = Image.new("RGB", (224, 224), (127, 127, 127))
    features = [
        {
            "image": img,
            "problem": "What color is the bird's beak?",
            "solution": "orange",
            "bbox": "[0.12, 0.34, 0.55, 0.78]",
        },
        {
            "image": img,
            "problem": "Read the title at the top.",
            "solution": "Annual Report",
            "bbox": "[0.05, 0.02, 0.95, 0.12]",
        },
    ]
    out = collator(features)

    # Group expansion: 2 prompts * group_size -> 6 contiguous rows.
    assert out["group_ids"].tolist() == [0, 0, 0, 1, 1, 1], out["group_ids"]
    assert len(out["locate_gt_boxes"]) == 2 * group_size
    assert out["locate_gt_boxes"][0] is not None and len(out["locate_gt_boxes"][0]) == 4
    assert out["student_prompt_input_ids"].shape[0] == 2 * group_size

    s_text = out["student_prompt_texts"][0]
    t_text = out["teacher_prompt_texts"][0]
    # Student: asked to locate a <box>, but NEVER given the hint/coordinates.
    assert "<box>" in s_text, "student prompt lost the locate instruction"
    assert "Hint:" not in s_text and "bounding box" not in s_text, "hint leaked to student!"
    # Teacher: silent hint with the GT coords, but NOT told to locate.
    assert "bounding box" in t_text and "[0.12, 0.34, 0.55, 0.78]" in t_text, t_text
    assert "<box>" not in t_text, "teacher must not be told to emit a <box>"

    # All group replicas of a prompt are identical (same length).
    s_mask = out["student_prompt_attention_mask"]
    g0 = {int(s_mask[i].sum()) for i in range(group_size)}
    assert len(g0) == 1, f"replicas of prompt 0 differ in length: {g0}"

    print(
        f"[log-sanity] check_collator OK  "
        f"(2 prompts x{group_size} -> {out['student_prompt_input_ids'].shape[0]} rows; "
        f"student has <box>, no hint; teacher has hint, no <box>)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=None,
        help="Student model dir/id for the full collator check (needs the processor).",
    )
    args = ap.parse_args()
    check_box_parsing()
    check_iou()
    check_advantage()
    check_pg_gradient()
    check_prompts()
    if args.model:
        check_collator(args.model)
    else:
        print("[log-sanity] skipped check_collator (pass --model to run it)")


if __name__ == "__main__":
    main()
