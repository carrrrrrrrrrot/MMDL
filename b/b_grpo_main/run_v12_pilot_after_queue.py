"""Run a four-step v1.2 GRPO GPU integration pilot after the active screen ends.

The synthetic rows verify the runtime path only; this pilot is not a model
quality experiment and must not be reported as common-parent GRPO.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def gpu_used_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(output.splitlines()[0].strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-state", type=Path, required=True)
    parser.add_argument("--screen-handoff", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--train-script", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--reward-policy", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    args = parser.parse_args()
    for path in (args.queue_state, args.python, args.train_script,
                 args.dataset_manifest, args.reward_policy, args.base_model):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    command = [
        str(args.python), str(args.train_script),
        "--dataset-manifest", str(args.dataset_manifest),
        "--reward-policy", str(args.reward_policy),
        "--parent-kind", "base", "--parent-model", str(args.base_model),
        "--output-dir", str(args.output_dir), "--max-steps", "4",
    ]
    subprocess.run(command + ["--validate-only"], check=True)
    deadline = time.monotonic() + args.timeout_seconds
    last_stage = None
    while time.monotonic() < deadline:
        state = json.loads(args.queue_state.read_text(encoding="utf-8"))
        stage = state.get("stage")
        if stage != last_stage:
            print(f"Queue stage: {stage}", flush=True)
            last_stage = stage
        if stage == "complete_exploratory_one_seed":
            if not args.screen_handoff.is_file():
                raise RuntimeError("Queue reports complete without screen handoff")
            if gpu_used_mib() <= 4096:
                print("Screen complete; starting v1.2 four-step GPU pilot", flush=True)
                subprocess.run(command, check=True)
                manifest = json.loads((args.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
                if manifest.get("status") != "complete" or not manifest.get("adapter_sha256"):
                    raise RuntimeError("v1.2 pilot output failed completion/adapter checks")
                print(json.dumps({"status": "complete", "run_manifest": str(args.output_dir / "run_manifest.json"),
                                  "adapter_sha256": manifest["adapter_sha256"]}, indent=2), flush=True)
                return
        elif stage in {"screen_blocked_baseline_mismatch", "failed"}:
            raise RuntimeError(f"Queue stopped before a completed screen: {stage}")
        time.sleep(args.poll_seconds)
    raise TimeoutError("Queue screen or GPU idle gate did not finish before timeout")


if __name__ == "__main__":
    main()
