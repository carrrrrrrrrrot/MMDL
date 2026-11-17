"""Terminal progress display for the B 6000 MMMU baseline (read only)."""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime
from pathlib import Path


RUN = Path("/data/chaewon/mmdl_eval_runs/b6000_baseline_v12_20261117")
SEEDS = ("3407", "1", "2")
TOTAL = 900


def _tail(path: Path, size: int = 512_000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - size))
        return stream.read().decode("utf-8", errors="replace")


def _status(seed: str) -> tuple[str, int]:
    root = RUN / f"seed_{seed}"
    meta = (root / "run_meta.txt").read_text(errors="replace") if (root / "run_meta.txt").is_file() else ""
    match = re.search(r"^exit_status:\s*(\d+)\s*$", meta, re.MULTILINE)
    if match:
        return ("완료" if match.group(1) == "0" else "실패"), (TOTAL if match.group(1) == "0" else 0)
    log = _tail(root / "stdout.log")
    numbers = re.findall(r"Processed prompts:[^\r\n]*?(\d+)/900", log)
    if numbers:
        return "진행 중", min(TOTAL, int(numbers[-1]))
    return ("시작 중", 0) if root.exists() else ("대기", 0)


def _bar(count: int, width: int = 30) -> str:
    filled = count * width // TOTAL
    return "█" * filled + "░" * (width - filled)


def render() -> bool:
    print("\x1b[2J\x1b[H", end="")
    print(f"B 서버 MMMU v1.2 기준선  |  {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}")
    print("각 시드 900문항 · 같은 모델 revision · 6000 측정\n")
    states = []
    for seed in SEEDS:
        state, count = _status(seed)
        states.append(state)
        print(f"seed {seed:>4}  [{_bar(count)}] {count:>3}/{TOTAL}  {state}")
    comparison = RUN / "comparison" / "comparison_summary.json"
    print("\n문항별 비교:", "완료" if comparison.is_file() else "대기")
    print("\n이 화면은 읽기 전용입니다. Ctrl+C는 화면만 닫고 평가 작업은 계속됩니다.")
    if comparison.is_file():
        print(f"비교 결과: {comparison}")
    return all(state == "완료" for state in states) and comparison.is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()
    while True:
        complete = render()
        if complete or args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
