"""Minimal isolated EasyR1 dependency and FlashAttention GPU probe."""

import json

import torch
from flash_attn import flash_attn_func

import hydra  # noqa: F401
import omegaconf  # noqa: F401
import ray  # noqa: F401
import tensordict  # noqa: F401
import vllm  # noqa: F401


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    query = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    output = flash_attn_func(query, query, query, causal=True)
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "flash_output_shape": list(output.shape),
                "flash_finite": bool(torch.isfinite(output).all()),
            }
        )
    )


if __name__ == "__main__":
    main()
