"""CPU integration check for A's exact ECD converter and B's GRPO loader.

Usage: python test_ecd_integration.py --a-converter TEAM_SHARE/convert_ecd.py
       --image /path/to/existing_image.png --processor /path/to/base_snapshot
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

from convert_approved import Approval, convert_row, verify_rendered


def import_converter(path: Path):
    spec = importlib.util.spec_from_file_location("a_convert_ecd", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import A ECD converter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-converter", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--processor", required=True, type=Path)
    args = parser.parse_args()
    a_converter = import_converter(args.a_converter.resolve(strict=True))
    image = args.image.resolve(strict=True)
    image_root = image.parent
    image_id = image.stem
    processor_path = args.processor.resolve(strict=True)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(processor_path)
    finals = ["Between 1550 and 1600", "Purple; approximately 0.015", "10"]
    rows = []
    for index, final in enumerate(finals):
        source = {
            "image_id": image_id,
            "question": f"For chart example {index}, what is the marked value?",
            "answer": (
                "I inspect the chart carefully, compare the labels and plotted values, "
                "then verify the marked value against the visible scale before deciding. "
                f"Answer: {final}"
            ),
            "split": "reasoning",
        }
        converted = a_converter.convert(index, source, str(image_root))
        if converted["status"] != "ok":
            raise AssertionError((index, converted["status"]))
        approval = Approval(converted["source"], converted["id"], "train")
        row, reason, images = convert_row(converted, approval, image_root, "ECD-test-284489f")
        if reason is not None or row is None:
            raise AssertionError((index, final, reason))
        if row["answer"] != final or row["answer_type"] != "open":
            raise AssertionError((index, row["answer"], row["answer_type"]))
        verify_rendered(converted["messages"], row, images, processor)
        rows.append(row)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b_grpo_main"))
    from data import load_manifest

    with tempfile.TemporaryDirectory() as temporary:
        manifest = Path(temporary) / "ecd_manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        loaded, summary = load_manifest(manifest)
    if len(loaded) != len(finals) or summary["answer_types"] != {"open": len(finals)}:
        raise AssertionError(summary)
    if [row["answer"] for row in loaded] != finals or any(row["options"] for row in loaded):
        raise AssertionError("B loader changed ECD final answers or option structure")
    print(f"PASS: exact A ECD converter → B approval adapter → B loader; {len(finals)}/{len(finals)} rendered prompts")


if __name__ == "__main__":
    main()
