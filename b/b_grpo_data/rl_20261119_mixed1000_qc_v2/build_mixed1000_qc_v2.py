#!/usr/bin/env python3
"""Replace three suspect CLEVR labels in the pinned mixed1000 v1 input.

The ECD rows and all heldout rows stay byte-equivalent. Replacement CLEVR
rows are rendered from A's pinned handoff, with C's SFT1 list checked by idx.
This script does not imply full visual-gold approval.
"""

import argparse
import hashlib
import json
import posixpath
import re
from pathlib import Path


EXPECTED_SHA = {
    "prior": "e71df0141dabc9d696cf67e5b47d35b05801633d145d855ca83b7efdbb95637e",
    "handoff": "686114a5a0df098c624e65ef56218ef7916aa7ab1d79bd5308cf2d722ce64e55",
    "c_sft1": "c950c5072b6ca77ed9a23acbf040ef02b0445ca95adf2e467a1e127d773be0e9",
}
EXCLUDE = (
    "mammoth_5933982",
    "mammoth_8467823",
    "mammoth_7615917",
)
ADD = {
    "mammoth_9365435": "4c8a344b45f12a10ee65ddb9fcf3ff80c91b46f65c059992d6cf4268918e987b",
    "mammoth_9456721": "b80824471aa3489c8c6ddd75d7c953774b87061d9ba6290fad6b6b19c96de8a5",
    "mammoth_5001946": "c7bedfb355ad38993d50eb61473f3cbf96d1aaea6f0a829bc4d6bf9c7462725c",
}
IMAGE_ROOT = "/data/chaewon/mmdl_grpo_data/mammoth_grpo_for_b_verified_20261118/mammoth_grpo_for_b"
A_COMMIT = "a61adb899d3c2274ba98cba7a360c5b4f333979f"
INSTRUCTION = 'Think step by step, then give your final answer on the last line in the form "Answer: <answer>".'


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def render(raw):
    question = raw["question"].strip()
    image = re.sub(r"/+", "/", raw["image"])
    sample_id = "a_" + hashlib.sha256(f"{raw['source']}\0{raw['id']}".encode()).hexdigest()[:20]
    return {
        "sample_id": sample_id,
        "source": raw["source"],
        "dataset_revision": "mammoth-vl-shard1-a61adb899d3c",
        "split": "train",
        "original_id": raw["id"],
        "image_paths": [posixpath.join(IMAGE_ROOT, image)],
        "image_sha256": raw["image_sha256"],
        "image_bytes": raw["image_bytes"],
        "question": question,
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Question: "},
                {"type": "image"},
                {"type": "text", "text": f"\n{question}\n{INSTRUCTION}"},
            ]},
        ],
        "answer": raw["answer"].strip(),
        "answer_type": "open",
        "choices": {},
        "a_answer_type": raw["answer_type"],
        "a_idx": raw["idx"],
        "a_used_in": raw["used_in"],
        "a_base_k_correct": raw["base_k_correct"],
        "a_text_only_suspect": raw["text_only_suspect"],
        "a_part": raw["part"],
        "a_handoff_commit": A_COMMIT,
        "a_used_in_dpo_pair": "dpo_pairs" in raw["used_in"],
        "a_used_in_dpo_pool": "dpo_pool" in raw["used_in"],
        "c_leakage_audit_status": "complete_within_documented_scope",
        "c_leakage_result": "not_detected_within_scope",
        "c_leakage_scope": "SFT1_12278_final",
        "c_audit_evidence_sha256": "23b7e04093f1730ae89de5770bc951c6281d24b803e738382de1932564fca0b7",
        "c_training_approval": False,
        "b_provisional_quality_screen": "CLEVR_source_spotcheck_only",
        "visual_gold_quality_verified": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "handoff", "c-sft1", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    inputs = {"prior": args.prior, "handoff": args.handoff, "c_sft1": args.c_sft1}
    for name, path in inputs.items():
        observed = sha(path)
        if observed != EXPECTED_SHA[name]:
            raise ValueError(f"{name} SHA256 {observed} differs from pinned {EXPECTED_SHA[name]}")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.parent.is_dir():
        raise FileNotFoundError(args.output.parent)

    prior = load(args.prior)
    if len(prior) != 1100:
        raise ValueError("Expected prior 1000 train + 100 heldout")
    if {row["original_id"] for row in prior if row["original_id"] in EXCLUDE} != set(EXCLUDE):
        raise ValueError("All excluded IDs must be present")
    if any(row["original_id"] in ADD for row in prior):
        raise ValueError("A replacement ID already occurs in prior manifest")
    raw = {row["id"]: row for row in load(args.handoff) if row["id"] in ADD}
    c_sft1 = {row["idx"]: row for row in load(args.c_sft1) if row.get("idx") in {x["idx"] for x in raw.values()}}
    if set(raw) != set(ADD) or len(c_sft1) != 3:
        raise ValueError("Replacement A/C records missing")
    replacements = []
    for ident, expected_hash in ADD.items():
        a = raw[ident]
        c = c_sft1[a["idx"]]
        if (a["source"] != "clevr_700k_new.json" or a["image_sha256"] != expected_hash
                or a["answer"] != "0" or a["answer_type"] != "numeric"
                or "sft1" not in a["used_in"] or a["text_only_suspect"]
                or c.get("status") != "ok" or c.get("source") != a["source"]
                or not c.get("question", "").rstrip().endswith(a["question"].strip())):
            raise ValueError(f"Replacement lineage/quality check failed: {ident}")
        replacements.append(render(a))

    train = [row for row in prior if row["split"] == "train" and row["original_id"] not in EXCLUDE]
    heldout = [row for row in prior if row["split"] == "heldout"]
    train.extend(replacements)
    if len(train) != 1000 or len(heldout) != 100:
        raise ValueError("Split sizes changed")
    tag = "mmdl-b-mixed1000-v1-order\0"  # Preserve old relative order.
    train.sort(key=lambda row: (hashlib.sha256(f"{tag}3407\0{row['sample_id']}".encode()).hexdigest(), row["sample_id"]))
    output = train + heldout
    for field, key in (("sample_id", lambda x: x["sample_id"]),
                       ("original_id", lambda x: x["original_id"]),
                       ("image_sha256", lambda x: x["image_sha256"]),
                       ("image_path", lambda x: x["image_paths"][0])):
        values = [key(row) for row in output]
        if len(set(values)) != len(output):
            raise ValueError(f"Duplicate {field}")
    data = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in output)
    args.output.write_text(data, encoding="utf-8", newline="\n")
    report = {
        "schema": "b_rl_mixed1000_qc_v2",
        "prior_sha256": EXPECTED_SHA["prior"],
        "handoff_sha256": EXPECTED_SHA["handoff"],
        "c_sft1_sha256": EXPECTED_SHA["c_sft1"],
        "excluded_suspect_ids": list(EXCLUDE),
        "added_visual_spotcheck_ids": list(ADD),
        "added_image_sha256": ADD,
        "train": 1000, "heldout": 100,
        "ecd_train": sum(row["source"] == "ECD-reasoning" for row in train),
        "clevr_train": sum(row["source"] == "clevr_700k_new.json" for row in train),
        "ecd_heldout": sum(row["source"] == "ECD-reasoning" for row in heldout),
        "clevr_heldout": sum(row["source"] == "clevr_700k_new.json" for row in heldout),
        "clevr_sft1_overlap": sum(row["source"] == "clevr_700k_new.json" and "sft1" in row.get("a_used_in", []) for row in train),
        "clevr_dpo_overlap": sum(row["source"] == "clevr_700k_new.json" and any(x in row.get("a_used_in", []) for x in ("dpo_pool", "dpo_pairs")) for row in train),
        "manifest_sha256": sha(args.output),
        "note": "Three suspicious visual golds excluded; three visually checked A SFT1 CLEVR rows added. Full C gold approval absent.",
    }
    report_path = args.output.with_name("selection_report.json")
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
