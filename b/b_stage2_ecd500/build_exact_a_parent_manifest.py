"""Record an A-style DPO(e) merge only after its original weight hashes match.

This is a one-time bridge from A's MERGE_INFO.json to B's EasyR1 parent
manifest schema. It never changes model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXPECTED = {
    "model-00001-of-00002.safetensors": "1d9e148aa1a102407bf3dd89b7f5a8d2bfbe792922bee9db243c4b54d03d1b2b",
    "model-00002-of-00002.safetensors": "0c615cbfe078470dfdeb299976580391f8482fed83ac521ec90b3de331b7f0f5",
    "model.safetensors.index.json": "c321e968083cc64dbdc77b9cc17842396cf376399f4ff4bca6b0a001fd22afef",
}
A_ADAPTER_SHA = "7650bfbbf72bb4ced94a605a00775d59b94c4eef7ae4a6b9265e19f5b250c5aa"
A_HASH_SOURCE = "https://github.com/carrrrrrrrrrot/MMDL/blob/b08ac1f/dpo_e_trunc_for_b/eval_4090_s3407/merged_dpo_e_trunc_SHA256.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--previous-b-manifest", type=Path, required=True)
    args = parser.parse_args()

    model = args.model.resolve(strict=True)
    target = model / "model_manifest.json"
    if target.exists():
        raise FileExistsError(f"Refusing to replace {target}")
    previous = json.loads(args.previous_b_manifest.read_text(encoding="utf-8"))
    if (previous.get("status") != "complete" or
            previous.get("parent_kind") != "merged_base_dpo" or
            previous.get("adapters", [{}])[0].get("weights_sha256") != A_ADAPTER_SHA):
        raise ValueError("Previous B manifest does not identify A's original DPO(e) adapter")

    observed = {name: sha256(model / name) for name in EXPECTED}
    if observed != EXPECTED:
        raise ValueError(f"A's original 65.15% merged weights differ: {observed}")
    merge_info = json.loads((model / "MERGE_INFO.json").read_text(encoding="utf-8"))
    if (merge_info.get("adapter_sha256", {}).get("adapter_model.safetensors") != A_ADAPTER_SHA or
            merge_info.get("base_revision") != previous.get("base_revision")):
        raise ValueError("A MERGE_INFO adapter or base revision differs")

    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(model.iterdir()) if path.is_file()
    }
    manifest = dict(previous)
    manifest.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model),
        "merged_model_path": str(model),
        "merge_method": "A team_share/merge_lora.py, PEFT merge_and_unload(), BF16 CPU, 5GB shards",
        "files": files,
        "a_original_4090_match": {
            "source": A_HASH_SOURCE,
            "expected_sha256": EXPECTED,
            "observed_sha256": observed,
            "status": "exact_weight_shards_and_index_match",
        },
    })
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "model_path": str(model),
                      "expected_weight_hashes_match": True,
                      "manifest_path": str(target)}, indent=2))


if __name__ == "__main__":
    main()
