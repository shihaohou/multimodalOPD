"""Convert local Vision-SR1-47K into ms-swift GRPO format (JSONL + image files).

Each output line:
  {"messages": [{"role": "user", "content": "<image>\\n<problem>"}],
   "images": ["/abs/path/0000001.png"],
   "solution": "<answer>"}

`solution` is the verifiable ground truth; ms-swift passes it to the reward
function during GRPO (see reward_accuracy.py). Images are dumped to disk because
ms-swift expects image file paths in a custom JSONL dataset.

Run in EITHER env (only needs `datasets` + `pillow`):
  python baseline/teacher_grpo/prepare_vision_sr1.py \
    --output-jsonl /home/.../datasets/vision_sr1_swift/train.jsonl \
    --image-dir   /home/.../datasets/vision_sr1_swift/images \
    --limit 200            # smoke; omit for the full 47k
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-path",
        default="/home/web_server/antispam/project/houshihao/datasets/Vision-SR1-47K",
    )
    ap.add_argument("--split", default="train")
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--answer-field", default="answer")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(args.dataset_path, split=args.split)
    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for index in range(total):
            row = dataset[index]
            image = row.get("images", row.get("image"))
            if isinstance(image, list):
                image = image[0] if image else None
            problem = str(row.get("problem", "")).strip()
            answer = row.get(args.answer_field)
            if answer is None or not str(answer).strip():
                answer = row.get("solution", "")
            answer = str(answer).strip()
            if image is None or not problem or not answer:
                skipped += 1
                continue
            image_path = image_dir / f"{index:07d}.png"
            try:
                image.convert("RGB").save(image_path)
            except Exception:
                skipped += 1
                continue
            record = {
                "messages": [{"role": "user", "content": "<image>\n" + problem}],
                "images": [str(image_path)],
                "solution": answer,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} samples (skipped {skipped}) to {out_path}")
    print(f"Images under {image_dir}")


if __name__ == "__main__":
    main()
