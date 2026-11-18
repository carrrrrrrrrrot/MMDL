"""CPU check: C ECD selection -> A-style v1.2 -> B adapter -> GRPO loader.

python test_convert_c_ecd.py --a-converter TEAM_SHARE/convert_ecd.py \
  --sft1-list TEAM_SHARE/c_list_sft1.jsonl [--processor QWEN_SNAPSHOT]
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import sys
import tempfile
import uuid
from pathlib import Path

from convert_approved import approvals, convert_row, verify_rendered
from convert_c_ecd import build


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lJ0AAAAASUVORK5CYII="
)


def import_a(path: Path):
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("a_convert_ecd_for_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-converter", required=True, type=Path)
    parser.add_argument("--sft1-list", required=True, type=Path)
    parser.add_argument("--processor", type=Path)
    args = parser.parse_args()
    a_path = args.a_converter.resolve(strict=True)
    sft_path = args.sft1_list.resolve(strict=True)
    a = import_a(a_path)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        image_root = root
        image_dir = root / "images"
        image_dir.mkdir()
        chart = f"b_grpo_test_{uuid.uuid4().hex}"
        image = image_dir / f"{chart}.png"
        image.write_bytes(PNG_1X1)
        item = {
            "id": f"ecd_{chart}_c1",
            "question": "What is the marked value in this chart?",
            "final_answer": "10",
            "image_relative_path": f"images/{image.name}",
        }
        selection = root / "c_selection.jsonl"
        write_jsonl(selection, [item])
        converted, rejects, provenance = build(selection, sft_path, image_root, a_path)
        assert not rejects and len(converted) == 1, rejects
        assert len(provenance["sft1_list_sha256"]) == 64

        source = {
            "image_id": chart,
            "question": item["question"],
            "answer": (
                "I inspect the chart carefully and compare the visible labels with the plotted mark. "
                "The indicated point matches the scale, so I can determine the final value clearly. "
                "Answer: 10"
            ),
            "split": "reasoning",
        }
        a_row = a.convert(0, source, str(image_dir))
        assert a_row["status"] == "ok", a_row
        ours = converted[0]
        assert ours["source"] == a_row["source"]
        assert ours["messages"][0] == a_row["messages"][0]
        a_blocks = a_row["messages"][1]["content"]
        b_blocks = ours["messages"][1]["content"]
        assert len(a_blocks) == len(b_blocks) == 3
        assert a_blocks[0] == b_blocks[0] and a_blocks[2] == b_blocks[2]
        assert a_blocks[1]["type"] == b_blocks[1]["type"] == "image"
        assert ours["messages"][2]["content"][0]["text"] == "Answer: 10"

        approval = approvals(selection)[0]
        manifest_row, reason, images = convert_row(ours, approval, image_root, "ECD-test")
        assert reason is None and manifest_row is not None, reason
        assert manifest_row["answer"] == "10"
        if args.processor:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(args.processor.resolve(strict=True))
            verify_rendered(ours["messages"], manifest_row, images, processor)
            verify_rendered(a_row["messages"], manifest_row, images, processor)

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b_grpo_main"))
        from data import load_manifest

        manifest = root / "manifest.jsonl"
        write_jsonl(manifest, [manifest_row])
        loaded, summary = load_manifest(manifest)
        assert len(loaded) == 1 and summary["answer_types"] == {"open": 1}
        assert loaded[0]["answer"] == "10"

        fixture_sft = root / "sft_fixture.jsonl"
        write_jsonl(fixture_sft, [{"id": "sft_used", "dataset": "ecd", "image": f"images/{image.name}"}])
        reused, blocked, _ = build(selection, fixture_sft, image_root, a_path)
        assert not blocked and reused[0]["flags"]["used_in_sft1_chart"]
        assert not reused[0]["flags"]["used_in_sft1_id"]
        same_id = root / "same_sft_id.jsonl"
        write_jsonl(same_id, [dict(item, id="sft_used")])
        reused, blocked, _ = build(same_id, fixture_sft, image_root, a_path)
        assert not blocked and reused[0]["flags"]["used_in_sft1_id"]
        cases = [
            ([dict(item, selection_status="metadata_only_requires_image_gold_and_leakage_review")], "selection_not_approved"),
            ([item, dict(item, id="second")], "duplicate_chart"),
            ([dict(item, image_relative_path="../escape.png")], "image_path_traversal"),
            ([dict(item, image_relative_path="missing.png")], "missing_image"),
            ([dict(item, final_answer="x" * 41)], "long_final_answer"),
        ]
        empty_sft = root / "empty_sft.jsonl"
        write_jsonl(empty_sft, [])
        for index, (rows, expected) in enumerate(cases):
            candidate = root / f"invalid_{index}.jsonl"
            write_jsonl(candidate, rows)
            _, errors, _ = build(candidate, empty_sft, image_root, a_path)
            assert expected in [error["reason"] for error in errors], (expected, errors)

    print("PASS: A ECD prompt parity, C approval, SFT1 usage flags, and five rejection cases")


if __name__ == "__main__":
    main()
