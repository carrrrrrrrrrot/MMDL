#!/usr/bin/env node
/** Extend the ECD QC-v2 training split with a subset of C-audited SFT1 ECD.
 *
 * This is an exploratory data-size ablation, not an item-level gold review.
 * The original 500 train rows and 97 heldout rows are retained unchanged.
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";

const EXPECTED = {
  source: "7065fc36cf148741c85ecf06be7a61948081c9dc422ef11b1103262847451e5c",
  sft1: "43159c7b68aaa84fb67718582f488c34992bdd10d7ef758f3b2a230353c5e0de",
  qcV2: "c05d5962a4b028236c726c72246eb9e8378d2ee0d96c9db3d0912f108625f1ed",
  cFinal: "b82241747c1c0259a9b40f66e61f828bffd0882fec8c4f0dd47f9fa6f8c2b332",
  sft1Hashes: "9e0817c7aeccadb077161b95334236c4dfbb56e1432f9da8b85c43d24ca23acb",
};
const BAD_CHARTS = new Set(["000259", "008560", "003223", "004222", "002406", "000838"]);
const RISKY_WORDING = /\b(?:average|mean|median|approximate(?:ly)?|estimated?|estimating|roughly|about|around|nearly|almost|nearest|round(?:ed|ing)?|close to)\b/i;
const BARE_NUMBER = /^[+-]?(?:0|[1-9]\d*)(?:\.\d{1,2})?$/;
const SYSTEM = "You are a helpful assistant.";
const ANSWER_LINE = 'Think step by step, then give your final answer on the last line in the form "Answer: <answer>".';

function sha(input) {
  return createHash("sha256").update(input).digest("hex");
}
function readPinned(path, expected) {
  const body = readFileSync(path);
  const got = sha(body);
  if (got !== expected) throw new Error(`Input SHA256 mismatch: ${path}: ${got}`);
  return body.toString("utf8");
}
function rows(body) {
  return body.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}
function argsFrom(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 2) {
    if (!argv[i]?.startsWith("--") || !argv[i + 1]) throw new Error("Expected --key value pairs");
    out[argv[i].slice(2)] = argv[i + 1];
  }
  for (const key of ["source", "sft1", "sft1-hashes", "qc-v2", "c-final", "image-root", "output-dir"]) {
    if (!out[key]) throw new Error(`Missing --${key}`);
  }
  return out;
}
function imageIdFromOld(row) {
  const name = basename(row.image_paths?.[0] ?? "");
  if (!/^\d{6}\.png$/.test(name)) throw new Error(`Bad QC-v2 image: ${row.original_id}`);
  return name.slice(0, -4);
}
function newRow(source, imageRoot, sftSha, cImageSha) {
  const question = source.question.trim();
  const imagePath = join(imageRoot, `${source.image_id}.png`);
  const image = readFileSync(imagePath);
  if (image.length < 24 || !image.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) {
    throw new Error(`Invalid PNG header: ${imagePath}`);
  }
  const imageSha = sha(image);
  if (imageSha !== cImageSha) throw new Error(`Image SHA256 differs from C SFT1 audit: ${source.id}`);
  return {
    sample_id: "b_" + sha(`ECD\0${source.id}`).slice(0, 20),
    source: "ECD-reasoning",
    dataset_revision: "ECD-10k-Images-284489f",
    split: "train",
    original_id: source.id,
    image_paths: [imagePath],
    image_sha256: imageSha,
    question,
    prompt: [
      { role: "system", content: SYSTEM },
      { role: "user", content: [
        { type: "text", text: "Question: " },
        { type: "image" },
        { type: "text", text: `\n${question}\n${ANSWER_LINE}` },
      ] },
    ],
    answer: source.final_answer.trim(),
    answer_type: "open",
    choices: {},
    c_leakage_result: "not_detected_within_scope",
    c_audit_item_file_sha256: sftSha,
    c_audit_source: "SFT1_final_20261118_subset",
    c_training_approval: false,
    visual_gold_quality_verified: false,
    b_filter_version: "ecd_qc_v2_plus_sft1_audited",
    ecd_source_line: source.idx,
  };
}

function main() {
  const a = argsFrom(process.argv.slice(2));
  const outputDir = resolve(a["output-dir"]);
  if (existsSync(outputDir)) throw new Error(`Output already exists: ${outputDir}`);
  const source = rows(readPinned(a.source, EXPECTED.source));
  const sft1 = rows(readPinned(a.sft1, EXPECTED.sft1));
  const sft1Hashes = rows(readPinned(a["sft1-hashes"], EXPECTED.sft1Hashes));
  const old = rows(readPinned(a["qc-v2"], EXPECTED.qcV2));
  const cFinal = JSON.parse(readPinned(a["c-final"], EXPECTED.cFinal));
  if (source.length !== 62604 || sft1.length !== 12278 || sft1Hashes.length !== 12278 || old.length !== 597 ||
      cFinal.status !== "final" || cFinal.training_rows !== 12278 ||
      cFinal.additional_review_outcome !== "no_leakage_detected") {
    throw new Error("Pinned source or C audit shape changed");
  }
  const sftEcd = sft1.filter((x) => x.dataset === "ecd");
  const sftIds = new Set(sftEcd.map((x) => x.id));
  const cImageHashes = new Map(sft1Hashes.filter((x) => x.dataset === "ecd").map((x) => [x.id, x.hash?.sha256]));
  if (sftEcd.length !== 5000 || sftIds.size !== 5000) throw new Error("Expected 5000 unique SFT1 ECD IDs");
  if (cImageHashes.size !== 5000 || [...cImageHashes.values()].some((x) => !/^[0-9a-f]{64}$/.test(x))) {
    throw new Error("Expected 5000 C-audited ECD image hashes");
  }
  const oldTrain = old.filter((x) => x.split === "train");
  const oldHeld = old.filter((x) => x.split === "heldout");
  const oldCharts = new Set(old.map(imageIdFromOld));
  const oldIds = new Set(old.map((x) => x.original_id));
  if (oldTrain.length !== 500 || oldHeld.length !== 97 || oldCharts.size !== 597 || oldIds.size !== 597) {
    throw new Error("Unexpected QC-v2 split, duplicate ID, or duplicate chart");
  }
  const extras = source.filter((x) =>
    x.used_in_sft1_this_question === true && !x.excluded_reason && x.in_dpo_pool !== true &&
    BARE_NUMBER.test(x.final_answer ?? "") && !RISKY_WORDING.test(x.question ?? "") &&
    !BAD_CHARTS.has(String(x.image_id)) && !oldCharts.has(String(x.image_id)));
  const extraIds = new Set(extras.map((x) => x.id));
  const extraCharts = new Set(extras.map((x) => String(x.image_id)));
  if (extras.length !== 298 || extraIds.size !== 298 || extraCharts.size !== 298 ||
      extras.some((x) => !sftIds.has(x.id) || oldIds.has(x.id))) {
    throw new Error("SFT1 audited extension is not the expected 298 unique new items");
  }
  const imageRoot = resolve(a["image-root"]);
  if (!statSync(imageRoot).isDirectory()) throw new Error("Image root is not a directory");
  const ordered = extras.sort((x, y) => sha(`3407|${x.id}`).localeCompare(sha(`3407|${y.id}`)));
  const added = ordered.map((x) => newRow(x, imageRoot, EXPECTED.sft1, cImageHashes.get(x.id)));
  const combined = [...old, ...added];
  if (combined.filter((x) => x.split === "train").length !== 798 ||
      combined.filter((x) => x.split === "heldout").length !== 97) {
    throw new Error("Expected 798 train and 97 heldout");
  }
  const body = combined.map((x) => JSON.stringify(x)).join("\n") + "\n";
  const addedBody = added.map((x) => JSON.stringify({
    id: x.original_id, image_path: x.image_paths[0], image_sha256: x.image_sha256,
    answer: x.answer, c_audit_source: x.c_audit_source,
  })).join("\n") + "\n";
  const report = {
    status: "exploratory_unverified_gold_no_c_training_approval",
    old_train: 500, new_train: 298, total_train: 798, heldout_unchanged: 97,
    old_manifest_sha256: EXPECTED.qcV2,
    source_sha256: EXPECTED.source, c_sft1_manifest_sha256: EXPECTED.sft1,
    c_sft1_image_hashes_sha256: EXPECTED.sft1Hashes,
    c_sft1_final_summary_sha256: EXPECTED.cFinal,
    c_audit_scope: "SFT1 specified data; MMMU and MMMU-Pro leakage not detected within documented scope",
    new_item_gold_warning: "ECD source gold has known errors; 298 new items are not individually gold verified",
    manifest_sha256: sha(body), added_items_sha256: sha(addedBody),
    image_root: imageRoot,
  };
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, "ecd_manifest.jsonl"), body, { flag: "wx" });
  writeFileSync(join(outputDir, "added_sft1_ecd.jsonl"), addedBody, { flag: "wx" });
  writeFileSync(join(outputDir, "selection_report.json"), JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(JSON.stringify(report, null, 2));
}

main();
