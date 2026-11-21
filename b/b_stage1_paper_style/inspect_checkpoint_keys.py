"""Read checkpoint key metadata without loading the full model into RAM."""

from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    lora = [(key, list(value.shape)) for key, value in state.items() if "lora_" in key]
    print(json.dumps({
        "total_keys": len(state),
        "first_keys": list(state)[:8],
        "lora_count": len(lora),
        "first_lora_keys": lora[:8],
    }, indent=2))


if __name__ == "__main__":
    main()
