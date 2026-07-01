"""Strip training-only Evidence Anchor OPD weights from saved checkpoints."""

from __future__ import annotations

import argparse

from baseline.anchor.opd_anchor_trainer import strip_anchor_auxiliary_weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove opd_anchor_projectors.* keys from Evidence Anchor OPD checkpoints "
            "so vLLM can load them for inference."
        )
    )
    parser.add_argument("checkpoint_dirs", nargs="+", help="Checkpoint directories to clean.")
    args = parser.parse_args()

    for checkpoint_dir in args.checkpoint_dirs:
        removed = strip_anchor_auxiliary_weights(checkpoint_dir)
        if removed:
            print(f"[strip-anchor] {checkpoint_dir}: removed {', '.join(removed)}")
        else:
            print(f"[strip-anchor] {checkpoint_dir}: no anchor auxiliary weights found")


if __name__ == "__main__":
    main()
