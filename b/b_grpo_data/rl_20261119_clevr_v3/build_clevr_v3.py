#!/usr/bin/env python3
"""Rebuild B's provisional CLEVR-only 500/100 GRPO manifest.

Requires A's exact handoff and C's exact leakage-audit inputs. This selects
rows deterministically; it does not confer C quality or training approval.
The output path must be new, so a prior experimental manifest is preserved.
"""

import argparse
import hashlib
import json
import posixpath
import re
from collections import Counter
from pathlib import Path


EXPECTED = {
    "handoff": "686114a5a0df098c624e65ef56218ef7916aa7ab1d79bd5308cf2d722ce64e55",
    "new_audit": "008d9e3a2bba1ed58d9bd9693d0af6b589a390634a0d35b42fd7406e27189016",
    "sft1_list": "43159c7b68aaa84fb67718582f488c34992bdd10d7ef758f3b2a230353c5e0de",
    "sft1_hashes": "9e0817c7aeccadb077161b95334236c4dfbb56e1432f9da8b85c43d24ca23acb",
    "sft1_report": "23b7e04093f1730ae89de5770bc951c6281d24b803e738382de1932564fca0b7",
}
A_COMMIT = "a61adb899d3c2274ba98cba7a360c5b4f333979f"
IMAGE_ROOT = "/data/chaewon/mmdl_grpo_data/mammoth_grpo_for_b_verified_20261118/mammoth_grpo_for_b"
INSTRUCTION = 'Think step by step, then give your final answer on the last line in the form "Answer: <answer>".'
EXCLUDED = {
    "mammoth_10014503": "prior CLEVR sample: red object count/gold uncertain",
    "mammoth_5643201": "prior CLEVR sample: material count/gold uncertain",
    "mammoth_9634542": "2026-11-19 CLEVR sample: behind relation/gold uncertain",
    "mammoth_6024101": "2026-11-19 CLEVR sample: left/in-front relation/gold uncertain",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def rank(tag, ident):
    return hashlib.sha256(f"{tag}|3407|{ident}".encode()).hexdigest()


def count(items, key):
    return dict(Counter(key(x) for x in items))


def render(raw, split):
    sft = "sft1" in raw["used_in"]
    question = raw["question"].strip()
    image = re.sub(r"/+", "/", raw["image"])
    source, ident = raw["source"], raw["id"]
    sample_id = "a_" + hashlib.sha256(f"{source}\0{ident}".encode()).hexdigest()[:20]
    return {
        "sample_id": sample_id,
        "source": source,
        "dataset_revision": "mammoth-vl-shard1-a61adb899d3c",
        "split": split,
        "original_id": ident,
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
        "c_leakage_scope": "SFT1_12278_final" if sft else "Audit1700_MAmmoTH900",
        "c_audit_evidence_sha256": EXPECTED["sft1_report"] if sft else EXPECTED["new_audit"],
        "c_training_approval": False,
        "b_provisional_quality_screen": "CLEVR_source_spotcheck_only",
        "visual_gold_quality_verified": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in EXPECTED:
        p.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    inputs = {name: getattr(args, name) for name in EXPECTED}
    for name, file in inputs.items():
        if digest(file) != EXPECTED[name]:
            raise ValueError(f"{name} SHA256 differs from frozen input")
    if args.output.exists():
        raise FileExistsError("Use a fresh output path")

    handoff = rows(inputs["handoff"])
    original = [x for x in handoff if x["source"] == "clevr_700k_new.json"]
    new_audit = {x["id"]: x for x in rows(inputs["new_audit"]) if x["dataset"] == "mammoth"}
    sft_list = {x["idx"]: x for x in rows(inputs["sft1_list"]) if x["dataset"] == "mammoth"}
    sft_hashes = {x["idx"]: x for x in rows(inputs["sft1_hashes"]) if x["dataset"] == "mammoth"}
    candidates = []
    seen_ids, seen_images, seen_questions = set(), set(), set()
    for x in original:
        if x["id"] in EXCLUDED:
            continue
        if x["id"] in seen_ids or x["image_sha256"] in seen_images or x["question"] in seen_questions:
            raise ValueError(f"Duplicate {x['id']}")
        seen_ids.add(x["id"]); seen_images.add(x["image_sha256"]); seen_questions.add(x["question"])
        if (x["text_only_suspect"] or x["answer_type"] not in ("numeric", "word") or
                x["options"] is not None or not x["question"].strip() or
                re.search(r"<image\s*\d*>|^\s*Answer\s*:", x["question"], re.I | re.M) or
                not x["answer"].strip() or len(x["answer"]) > 40):
            raise ValueError(f"Format {x['id']}")
        if "sft1" in x["used_in"]:
            c, h = sft_list.get(x["idx"]), sft_hashes.get(x["idx"])
            if not c or not h or c["source"] != x["source"] or h["hash"]["sha256"] != x["image_sha256"]:
                raise ValueError(f"SFT1 lineage {x['id']}")
        else:
            c = new_audit.get(x["id"])
            if (not c or c["leakage_audit_status"] != "complete_within_documented_scope" or
                    c["leakage_result"] != "not_detected_within_scope" or
                    c["image_sha256"] != x["image_sha256"]):
                raise ValueError(f"New900 C audit {x['id']}")
        candidates.append(x)
    sft = [x for x in candidates if "sft1" in x["used_in"]]
    fresh = [x for x in candidates if "sft1" not in x["used_in"]]
    if len(original) != 664 or len(fresh) != 479 or len(sft) != 181:
        raise ValueError("Unexpected CLEVR source counts")

    held_zero = sorted((x for x in fresh if x["base_k_correct"] == 4 and x["answer"] == "0"),
                       key=lambda x: rank("heldout-zero", x["id"]))[:50]
    held_nonzero = sorted((x for x in fresh if x["base_k_correct"] == 4 and x["answer"] != "0"),
                          key=lambda x: rank("heldout-nonzero", x["id"]))[:50]
    held = held_zero + held_nonzero
    held_ids = {x["id"] for x in held}
    fresh_train = [x for x in fresh if x["id"] not in held_ids]
    sft_train = sorted(sft, key=lambda x: (x["answer"] == "0", rank("sft-train", x["id"])))[:121]
    train = sorted(fresh_train + sft_train, key=lambda x: rank("train-order", x["id"]))
    output = [render(x, "train") for x in train] + [render(x, "heldout") for x in held]
    if len(train) != 500 or len(held) != 100:
        raise ValueError("Wrong split count")
    for key in ("sample_id", "image_sha256", "question"):
        if len({x[key] for x in output}) != 600:
            raise ValueError(f"Output duplicate {key}")
    args.output.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in output), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "sha256": digest(args.output),
        "train": len(train), "heldout": len(held),
        "train_k": count(output[:500], lambda x: str(x["a_base_k_correct"])),
        "train_answer_zero": sum(x["answer"] == "0" for x in output[:500]),
        "heldout_k4": sum(x["a_base_k_correct"] == 4 for x in output[500:]),
        "c_training_approval": False,
    }, indent=2))


if __name__ == "__main__":
    main()
