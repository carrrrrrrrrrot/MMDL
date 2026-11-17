"""Terminal dashboard for the B baseline follow-up and common-parent gate."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


DEFAULT_QUEUE = Path("/data/chaewon/mmdl_b_eval_ops/orchestrations/base_insurance_001/state.json")
DEFAULT_QUEUE_LOG = Path("/data/chaewon/mmdl_b_eval_ops/base_insurance_001.queue.log")
DEFAULT_PILOT = Path("/data/chaewon/mmdl_grpo_runs/v12_gpu_pilot_002/run_manifest.json")
DEFAULT_PILOT_LOG = Path("/data/chaewon/mmdl_grpo_runs/v12_gpu_pilot_002.launch.log")
DEFAULT_CONFIG = Path("/data/chaewon/mmdl_handoff/common_parent_config.json")
DEFAULT_HANDOFF = Path("/data/chaewon/mmdl_handoff/common_parent_001/state.json")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def tail(path: Path, size: int = 40000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def bar(done: int, total: int, width: int = 24) -> str:
    filled = round(width * min(done, total) / total)
    return "[" + "#" * filled + "." * (width - filled) + f"] {done}/{total}"


def screen_progress(log: str) -> tuple[int, int] | None:
    matches = re.findall(r"Processed prompts:[^\r\n]*?\b(\d+)/(\d+)\b", log)
    if not matches:
        return None
    return tuple(int(value) for value in matches[-1])


def pilot_progress(log: str) -> tuple[int, int] | None:
    matches = re.findall(r"\b(\d+)/(\d+)\s*\[[^\r\n]*?\]", log)
    if not matches:
        return None
    return tuple(int(value) for value in matches[-1])


def snapshot(args: argparse.Namespace) -> str:
    now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%m-%d %H:%M:%S KST")
    queue = read_json(args.queue)
    queue_stage = queue.get("stage", "unknown")
    queue_text = f"합성 base→GRPO 확인: {queue_stage}"
    if queue_stage == "exploratory_screen_900_seed_3407":
        progress = screen_progress(tail(args.queue_log))
        queue_text += "  " + (bar(*progress) if progress else "900문항 평가 시작 중")
    pilot = read_json(args.pilot)
    if pilot.get("status") == "complete":
        pilot_text = "v1.2 보상 GPU 파일럿: 완료"
    elif pilot:
        pilot_text = f"v1.2 보상 GPU 파일럿: {pilot.get('status', '상태 확인 중')}"
        if pilot.get("status") == "running":
            progress = pilot_progress(tail(args.pilot_log))
            if progress:
                pilot_text += "  " + bar(*progress)
    else:
        log = tail(args.pilot_log, 3000)
        pilot_text = ("v1.2 보상 GPU 파일럿: 앞선 900문항 평가 완료 대기"
                      if "Queue stage:" in log else "v1.2 보상 GPU 파일럿: 예약 상태 확인 중")
    handoff = read_json(args.handoff)
    if handoff:
        handoff_text = f"실데이터 공통 부모: {handoff.get('status', 'unknown')}"
    else:
        config = read_json(args.config)
        required = ("sft_adapter", "sft_adapter_sha256", "a_converted_jsonl",
                    "c_approved_ids", "c_clearance_json", "image_root",
                    "dataset_revision", "max_steps", "min_train_rows")
        missing = [name for name in required if not config.get(name)]
        handoff_text = ("실데이터 공통 부모: A·C 입력 대기 (" + ", ".join(missing) + ")"
                        if missing else "실데이터 공통 부모: 입력 검증 대기")
    return "\n".join((now, queue_text, pilot_text, handoff_text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--queue-log", type=Path, default=DEFAULT_QUEUE_LOG)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--pilot-log", type=Path, default=DEFAULT_PILOT_LOG)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be positive")
    while True:
        if not args.once:
            print("\033[2J\033[H", end="")
        print(snapshot(args), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
