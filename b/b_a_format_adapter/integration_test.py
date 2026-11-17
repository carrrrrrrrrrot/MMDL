"""CPU smoke using the actual A convert_v12.py and Qwen processor."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-converter", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--processor", required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("a_convert_v12", args.a_converter)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import A converter")
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    source = "geo170k(qa)"
    records = [
        {
            "id": "B-A-MC-1", "source": source, "image": str(args.image.resolve(strict=True)), "n_turns": 1,
            "conversations": [
                {"from": "human", "value": "<image>\nHow many red squares are visible?\nChoices:\nA: one\nB: two\nC: three\nD: four"},
                {"from": "gpt", "value": "I inspect the red cells in the figure carefully and count each distinct square once. There are two red squares.\nAnswer:B"},
            ],
        },
        {
            "id": "B-A-OPEN-1", "source": source, "image": str(args.image.resolve(strict=True)), "n_turns": 1,
            "conversations": [
                {"from": "human", "value": "<image>\nHow many red squares are visible?"},
                {"from": "gpt", "value": "I inspect the red cells in the figure carefully and count each distinct square once. There are two red squares. The answer is 2"},
            ],
        },
    ]
    converted = [a.convert(record) for record in records]
    if any(row["status"] != "ok" for row in converted):
        raise AssertionError([row["status"] for row in converted])
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        converted_path = root / "a_converted.jsonl"
        converted_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted), encoding="utf-8")
        approvals = root / "approved.csv"
        with approvals.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["source", "id", "split"])
            writer.writeheader()
            writer.writerow({"source": source, "id": records[0]["id"], "split": "train"})
            writer.writerow({"source": source, "id": records[1]["id"], "split": "train"})
        output = root / "b_manifest.jsonl"
        command = [
            sys.executable, str(Path(__file__).with_name("convert_approved.py")),
            "--converted-jsonl", str(converted_path),
            "--approved-ids", str(approvals),
            "--image-root", str(root),
            "--dataset-revision", "test-shard-1",
            "--output", str(output),
            "--processor", str(args.processor),
            "--render-limit", "0",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        report = json.loads(output.with_suffix(".jsonl.report.json").read_text(encoding="utf-8"))
        if len(rows) != 2 or {row["answer_type"] for row in rows} != {"mc", "open"}:
            raise AssertionError(rows)
        if report["rendered_parity_count"] != 2 or report["structural_parity_count"] != 2:
            raise AssertionError(report)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b_grpo_main"))
        from data import load_manifest

        loaded, summary = load_manifest(output)
        if len(loaded) != 2 or summary["answer_types"] != {"mc": 1, "open": 1}:
            raise AssertionError(summary)
        mc = next(row for row in loaded if row["answer_type"] == "mc")
        if mc["options"] != ["one", "two", "three", "four"]:
            raise AssertionError(mc["options"])
    print("PASS: A converter → C approval → B loader, 2/2 exact rendered prompts")


if __name__ == "__main__":
    main()
