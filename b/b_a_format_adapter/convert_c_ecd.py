"""Convert C-approved ECD JSONL to A-style v1.2 records for B's adapter.

The original C JSONL remains the --approved-ids input to convert_approved.py.
This bridge validates one question per chart and excludes A's SFT1 charts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def a_prompts(converter: Path) -> tuple[str, str, str, Path]:
    """Load A's shared constants, failing if that converter changes its contract."""
    converter = converter.resolve(strict=True)
    sys.path.insert(0, str(converter.parent))
    try:
        spec = importlib.util.spec_from_file_location("a_shared_convert_ecd", converter)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load A converter: {converter}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        values = (module.SYSTEM_PROMPT, module.PRE_PROMPT, module.OPEN_PROMPT)
    finally:
        sys.path.pop(0)
    if values[0] != "You are a helpful assistant." or values[1] != "Question: ":
        raise ValueError("A's v1.2 prompt contract changed")
    if 'Answer: <answer>' not in values[2]:
        raise ValueError("A's open-answer instruction changed")
    dependency = converter.with_name("convert_v12.py").resolve(strict=True)
    return (*values, dependency)


def _required_text(item: dict[str, Any], field: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing_or_invalid_{field}")
    return value.strip()


def _alias(item: dict[str, Any], primary: str, alternate: str) -> str:
    values = [(key, item[key]) for key in (primary, alternate) if key in item]
    if not values:
        raise ValueError(f"missing_{primary}")
    if len(values) == 2 and values[0][1] != values[1][1]:
        raise ValueError(f"conflicting_{primary}_{alternate}")
    value = values[0][1]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid_{primary}")
    return value.strip()


def _image(image_root: Path, relative: str) -> tuple[Path, str]:
    if "\\" in relative or ":" in relative:
        raise ValueError("image_path_not_posix_relative")
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts):
        raise ValueError("image_path_traversal")
    resolved = (image_root / Path(*path.parts)).resolve(strict=False)
    if not resolved.is_relative_to(image_root):
        raise ValueError("image_path_traversal")
    if not resolved.is_file():
        raise ValueError("missing_image")
    if resolved.suffix.lower() != ".png":
        raise ValueError("ecd_image_not_png")
    return resolved, resolved.stem.casefold()


def sft1_exclusions(path: Path) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    charts: set[str] = set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), (str, int)):
                raise ValueError(f"SFT1 list line {line_number} lacks id")
            ids.add(str(row["id"]))
            if str(row.get("dataset", "")).casefold() == "ecd":
                image = row.get("image")
                if not isinstance(image, str) or not image:
                    raise ValueError(f"SFT1 ECD line {line_number} lacks image")
                charts.add(PurePosixPath(image.replace("\\", "/")).stem.casefold())
    return ids, charts


def build(
    selection: Path, sft_list: Path, image_root: Path, a_converter: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    selection = selection.resolve(strict=True)
    sft_list = sft_list.resolve(strict=True)
    image_root = image_root.resolve(strict=True)
    system, pre_prompt, open_prompt, dependency = a_prompts(a_converter)
    sft_ids, sft_charts = sft1_exclusions(sft_list)
    seen_ids: set[str] = set()
    seen_charts: set[str] = set()
    records: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    with selection.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item: Any = None
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("row_not_object")
                raw_id = item.get("id", item.get("original_id"))
                if not isinstance(raw_id, (str, int)) or not str(raw_id).strip():
                    raise ValueError("missing_id")
                item_id = str(raw_id).strip()
                question = _required_text(item, "question")
                answer = _alias(item, "answer", "final_answer")
                relative = _alias(item, "image_relative_path", "image")
                if len(answer) > 40:
                    raise ValueError("long_final_answer")
                if re.search(r"(?i)\banswer\s*:", answer) or "\n" in answer:
                    raise ValueError("answer_must_be_final_only")
                if item.get("source") not in (None, "ECD", "ECD-reasoning"):
                    raise ValueError("source_must_be_ECD_or_ECD-reasoning_or_omitted")
                if item.get("split", "train") not in ("train", "heldout"):
                    raise ValueError("split_must_be_train_or_heldout")
                resolved, chart = _image(image_root, relative)
                if item_id in sft_ids:
                    raise ValueError("sft1_id_overlap")
                if chart in sft_charts:
                    raise ValueError("sft1_ecd_chart_overlap")
                if item_id in seen_ids:
                    raise ValueError("duplicate_id")
                if chart in seen_charts:
                    raise ValueError("duplicate_chart")
                seen_ids.add(item_id)
                seen_charts.add(chart)
                user = f"{pre_prompt}<image 1>\n{question}\n{open_prompt}"
                before, after = user.split("<image 1>")
                records.append({
                    "idx": line_number - 1,
                    "id": item_id,
                    "source": item.get("source") or "ECD-reasoning",
                    "question_type": "open",
                    "messages": [
                        {"role": "system", "content": [{"type": "text", "text": system}]},
                        {"role": "user", "content": [
                            {"type": "text", "text": before},
                            {"type": "image", "image": relative},
                            {"type": "text", "text": after},
                        ]},
                        {"role": "assistant", "content": [{"type": "text", "text": f"Answer: {answer}"}]},
                    ],
                    "images": [relative],
                    "status": "ok",
                    "flags": {"split": "reasoning", "c_approved_grpo_gold_only": True},
                })
            except (ValueError, json.JSONDecodeError) as exc:
                rejects.append({
                    "line": line_number,
                    "id": str(item.get("id", item.get("original_id", ""))) if isinstance(item, dict) else "",
                    "reason": str(exc),
                })
    provenance = {
        "selection_sha256": sha256(selection),
        "sft1_list_sha256": sha256(sft_list),
        "a_converter_sha256": sha256(a_converter),
        "a_convert_v12_sha256": sha256(dependency),
    }
    return records, rejects, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-jsonl", required=True, type=Path)
    parser.add_argument("--sft1-list", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--a-converter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("Choose fresh converted and report paths")
    records, rejects, provenance = build(
        args.selection_jsonl, args.sft1_list, args.image_root, args.a_converter
    )
    report: dict[str, Any] = {
        **provenance,
        "selected_count": len(records) + len(rejects),
        "accepted_count": len(records),
        "rejected_count": len(rejects),
        "rejected_by_reason": dict(Counter(row["reason"] for row in rejects)),
        "rejections": rejects,
        "output": str(args.output.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not rejects and records:
        with args.output.open("x", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["output_sha256"] = sha256(args.output)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rejections"}, ensure_ascii=False, indent=2))
    if rejects or not records:
        raise ValueError("C selection has rejected rows or is empty; see rejection report; no converted output written")


if __name__ == "__main__":
    main()
