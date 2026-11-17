"""Turn C-approved A convert_v12.py rows into B's GRPO JSONL manifest.

The adapter preserves A's system/user text and image block order. It never
selects data on its own: an explicit C-approved ID list is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SYSTEM = "You are a helpful assistant."
FINAL = re.compile(r"^Answer:\s*(\S.*?)\s*$")
MC_OPTION = re.compile(r"(?m)^([A-D])\.\s+(.+)$")


@dataclass(frozen=True)
class Approval:
    source: str | None
    original_id: str
    split: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def approvals(path: Path) -> list[Approval]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
    else:
        records = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                item = json.loads(line) if line.lstrip().startswith("{") else {"id": line.strip()}
                records.append(item)
    result: list[Approval] = []
    seen: set[tuple[str | None, str]] = set()
    for index, item in enumerate(records, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Approved IDs row {index} must be an object")
        raw_id = item.get("id", item.get("original_id"))
        source = item.get("source") or None
        split = item.get("split") or "train"
        if not isinstance(raw_id, (str, int)) or not str(raw_id).strip():
            raise ValueError(f"Approved IDs row {index} lacks id")
        if source is not None and not isinstance(source, str):
            raise ValueError(f"Approved IDs row {index} has invalid source")
        if split not in {"train", "heldout"}:
            raise ValueError(f"Approved IDs row {index} has unsupported split {split!r}")
        approved = Approval(source, str(raw_id), split)
        key = approved.source, approved.original_id
        if key in seen:
            raise ValueError(f"Duplicate approved ID {key}")
        seen.add(key)
        result.append(approved)
    if not result:
        raise ValueError("Approved ID list is empty")
    return result


def _text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        for block in content
    ):
        return "".join(block["text"] for block in content)
    return None


def _answer(messages: list[dict[str, Any]], kind: str) -> tuple[str | None, str | None]:
    if len(messages) < 3 or messages[2].get("role") != "assistant":
        return None, "missing_assistant"
    content = _text(messages[2].get("content"))
    if content is None:
        return None, "unsupported_assistant_content"
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return None, "empty_assistant"
    match = FINAL.fullmatch(lines[-1])
    if match is None:
        return None, "missing_final_answer_line"
    answer = match.group(1).strip()
    if kind == "multiple-choice":
        return (answer, None) if re.fullmatch(r"[A-D]", answer) else (None, "invalid_mc_gold")
    if kind != "open":
        return None, "unsupported_question_type"
    if len(answer) > 40:
        return None, "long_open_gold"
    return answer, None


def convert_row(
    rec: dict[str, Any], approval: Approval, image_root: Path, revision: str
) -> tuple[dict[str, Any] | None, str | None, list[Path]]:
    if rec.get("status") != "ok":
        return None, f"status_{rec.get('status', 'missing')}", []
    if rec.get("flags", {}).get("long_final_answer"):
        return None, "long_final_answer_flag", []
    messages = rec.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return None, "missing_messages", []
    system, user = messages[:2]
    if system.get("role") != "system" or _text(system.get("content")) != SYSTEM:
        return None, "system_prompt_mismatch", []
    if user.get("role") != "user" or not isinstance(user.get("content"), list):
        return None, "unsupported_user_content", []
    raw_images = rec.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        return None, "missing_images", []
    image_paths: list[Path] = []
    for item in raw_images:
        if not isinstance(item, str) or not item:
            return None, "non_path_image", []
        path = Path(item)
        if not path.is_absolute():
            path = image_root / path
        path = path.resolve(strict=False)
        if not path.is_file():
            return None, "missing_image_file", []
        image_paths.append(path)
    content: list[dict[str, str]] = []
    image_index = 0
    for block in user["content"]:
        if not isinstance(block, dict):
            return None, "invalid_content_block", []
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            content.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image":
            if image_index >= len(image_paths):
                return None, "image_block_count_mismatch", []
            original_payload = block.get("image")
            if original_payload is not None and str(original_payload) != str(raw_images[image_index]):
                return None, "image_block_payload_mismatch", []
            content.append({"type": "image"})
            image_index += 1
        else:
            return None, "invalid_content_block", []
    if image_index != len(image_paths):
        return None, "image_block_count_mismatch", []
    text_blocks = [block["text"] for block in content if block["type"] == "text"]
    user_text = "\n".join(text_blocks)
    if "<image" in user_text.lower():
        return None, "literal_image_marker_remains", []
    if not user_text.startswith("Question:"):
        return None, "question_prefix_missing", []
    if rec.get("question_type") == "multiple-choice":
        if 'Answer: X' not in user_text:
            return None, "mc_instruction_mismatch", []
        options = dict(MC_OPTION.findall(user_text))
        if len(options) < 2 or len(options) > 4:
            return None, "unsupported_option_count", []
        answer_type = "mc"
    elif rec.get("question_type") == "open":
        if 'Answer: <answer>' not in user_text:
            return None, "open_instruction_mismatch", []
        options = {}
        answer_type = "open"
    else:
        return None, "unsupported_question_type", []
    answer, error = _answer(messages, rec["question_type"])
    if error:
        return None, error, []
    if answer_type == "mc" and answer not in options:
        return None, "gold_option_missing", []
    end = user_text.find("\nOptions:")
    if end < 0:
        end = user_text.find("\nThink step by step")
    if end < 0:
        return None, "instruction_boundary_missing", []
    question = user_text[len("Question:"):end].strip()
    if not question or question not in user_text:
        return None, "question_empty_or_misaligned", []
    prompt = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]
    # Structural parity on every row: exact system/user text and image order.
    rebuilt = [
        {"role": "system", "content": [{"type": "text", "text": prompt[0]["content"]}]},
        {"role": "user", "content": [
            ({"type": "image", "image": raw_images[index]}
             if block["type"] == "image" else {"type": "text", "text": block["text"]})
            for index, block in enumerate_images(content)
        ]},
    ]
    if rebuilt != messages[:2]:
        return None, "structural_prompt_mismatch", []
    unique = hashlib.sha256(f"{rec.get('source')}\0{rec.get('id')}".encode()).hexdigest()[:20]
    row = {
        "sample_id": f"a_{unique}",
        "source": str(rec.get("source")),
        "dataset_revision": revision,
        "split": approval.split,
        "original_id": str(rec.get("id")),
        "image_paths": [str(path) for path in image_paths],
        "question": question,
        "prompt": prompt,
        "answer": answer,
        "answer_type": answer_type,
        "choices": options,
        "a_conversion_status": "ok",
        "a_flags": rec.get("flags", {}),
    }
    return row, None, image_paths


def enumerate_images(content: list[dict[str, str]]):
    image_index = 0
    for block in content:
        if block["type"] == "image":
            yield image_index, block
            image_index += 1
        else:
            yield -1, block


def verify_rendered(a_messages: list[dict[str, Any]], row: dict[str, Any], images: list[Path], processor) -> None:
    from PIL import Image
    from trl.data_utils import prepare_multimodal_messages

    loaded = []
    try:
        for path in images:
            loaded.append(Image.open(path).convert("RGB"))
        a_user = []
        index = 0
        for block in a_messages[1]["content"]:
            if block["type"] == "image":
                a_user.append({"type": "image", "image": loaded[index]})
                index += 1
            else:
                a_user.append({"type": "text", "text": block["text"]})
        source = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": a_user},
        ]
        target = prepare_multimodal_messages(row["prompt"], images=loaded)
        source_rendered = processor.apply_chat_template(source, add_generation_prompt=True, tokenize=False)
        target_rendered = processor.apply_chat_template(target, add_generation_prompt=True, tokenize=False)
        if source_rendered != target_rendered:
            raise ValueError(f"rendered prompt differs for {row['sample_id']}")
    finally:
        for image in loaded:
            image.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted-jsonl", required=True, type=Path)
    parser.add_argument("--approved-ids", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--processor", type=Path, help="Optional Qwen processor for rendered parity checks")
    parser.add_argument("--render-limit", type=int, default=20, help="0 checks all selected rows")
    args = parser.parse_args()
    source = args.converted_jsonl.resolve(strict=True)
    approved_path = args.approved_ids.resolve(strict=True)
    image_root = args.image_root.resolve(strict=True)
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".report.json").exists():
        raise FileExistsError("Choose fresh output and report paths")
    if not args.dataset_revision.strip() or args.render_limit < 0:
        raise ValueError("Revision must be non-empty and render-limit non-negative")
    selected = approvals(approved_path)
    by_key = {(item.source, item.original_id): item for item in selected}
    by_id: dict[str, list[Approval]] = {}
    for item in selected:
        by_id.setdefault(item.original_id, []).append(item)
    processor = None
    if args.processor:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.processor.resolve(strict=True))
    rows: list[dict[str, Any]] = []
    rejects: list[dict[str, str]] = []
    matched: Counter[Approval] = Counter()
    rendered_checks = 0
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError(f"Converted row {line_number} is not an object")
            rec_id = str(rec.get("id"))
            rec_source = rec.get("source")
            key = (rec_source, rec_id)
            approved = by_key.get(key) or by_key.get((None, rec_id))
            if approved is None:
                continue
            matched[approved] += 1
            if matched[approved] > 1:
                raise ValueError(f"Approved ID appears multiple times in A converted output: {key}")
            # An unqualified ID is acceptable only when it maps to one source.
            if approved.source is None and len(by_id[rec_id]) > 1:
                raise ValueError(f"Ambiguous unqualified approved ID: {rec_id}")
            row, reason, images = convert_row(rec, approved, image_root, args.dataset_revision)
            if reason:
                rejects.append({"source": str(rec_source), "id": rec_id, "reason": reason})
                continue
            assert row is not None
            if processor and (args.render_limit == 0 or rendered_checks < args.render_limit):
                verify_rendered(rec["messages"], row, images, processor)
                rendered_checks += 1
            rows.append(row)
    missing = [item for item in selected if matched[item] == 0]
    if missing:
        raise ValueError(f"{len(missing)} C-approved IDs were absent from A converted file: {missing[:5]}")
    if not rows:
        raise ValueError("No approved, scorable rows survived conversion")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Converted sample_id collision")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "source_jsonl": str(source),
        "source_sha256": sha256(source),
        "approved_ids": str(approved_path),
        "approved_sha256": sha256(approved_path),
        "dataset_revision": args.dataset_revision,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "approved_count": len(selected),
        "converted_count": len(rows),
        "rejected_count": len(rejects),
        "rejected_by_reason": dict(Counter(row["reason"] for row in rejects)),
        "rejections": rejects,
        "structural_parity_count": len(rows),
        "rendered_parity_count": rendered_checks,
        "answer_types": dict(Counter(row["answer_type"] for row in rows)),
    }
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rejections"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
