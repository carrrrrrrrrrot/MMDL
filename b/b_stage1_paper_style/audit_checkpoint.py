"""Print compact key statistics for an EasyR1 model checkpoint."""

import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    keys = list(state)
    lora = [key for key in keys if "lora_" in key]
    print(json.dumps({
        "total_keys": len(keys),
        "lora_keys": len(lora),
        "first_keys": keys[:3],
        "first_lora_keys": lora[:4],
        "last_lora_keys": lora[-4:],
        "lora_numel": sum(state[key].numel() for key in lora),
    }))


if __name__ == "__main__":
    main()
