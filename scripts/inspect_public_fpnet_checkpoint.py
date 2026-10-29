from __future__ import annotations

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT
    / "data/external/casmi26-public-assets/"
      "casmi26-fpnet-full1/fpnet_full1.pt"
)


def summarize(value, depth: int = 0):
    if depth > 2:
        return repr(type(value))

    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "model":
                result[key] = f"<state_dict with {len(item)} tensors>"
            else:
                result[key] = summarize(
                    item,
                    depth + 1,
                )
        return result

    if isinstance(value, (list, tuple)):
        if len(value) <= 100:
            return [
                summarize(
                    item,
                    depth + 1,
                )
                for item in value
            ]
        return (
            f"<{type(value).__name__} "
            f"len={len(value)}>"
        )

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            type(None),
        ),
    ):
        return value

    return repr(value)


def main() -> None:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    print("Checkpoint metadata")
    print("=" * 80)

    for key, value in checkpoint.items():
        if key == "model":
            continue
        print(
            key,
            "=",
            summarize(value),
        )

    state = checkpoint["model"]

    print()
    print("Relevant state-dict tensor shapes")
    print("=" * 80)

    prefixes = (
        "mz_emb.",
        "nl_emb.",
        "int_emb.",
        "pk.",
        "prec_emb.",
        "ce_emb.",
        "ad.",
        "ins.",
        "gl.",
        "elem_head.",
        "head.",
    )

    for key in sorted(state):
        if key.startswith(prefixes):
            tensor = state[key]
            print(
                f"{key:40s}",
                tuple(tensor.shape),
                tensor.dtype,
            )

    print()
    print("All state-dict keys")
    print("=" * 80)

    for key in sorted(state):
        tensor = state[key]
        print(
            f"{key:48s}",
            tuple(tensor.shape),
        )


if __name__ == "__main__":
    main()
