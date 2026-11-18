/** Select metadata-only ECD GRPO candidates for later visual and C review.
 *
 * This does not approve questions for training. It does not inspect image pixels,
 * verify the gold answer, check benchmark leakage, or generate model rollouts.
 */

import { createHash } from "node:crypto";
import { createReadStream, createWriteStream, existsSync, readFileSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { createInterface } from "node:readline";

function parseArgs(argv) {
  const args = { limit: 2000, seed: 3407, mode: "broad" };
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!value || !["--input", "--output", "--limit", "--seed", "--mode", "--exclude-ids"].includes(key)) {
      throw new Error("Usage: node select_ecd_candidates.mjs --input FILE --output FILE [--limit N] [--seed N] [--mode broad|priority_review] [--exclude-ids JSONL]");
    }
    args[key.slice(2)] = value;
  }
  args.limit = Number(args.limit);
  args.seed = Number(args.seed);
  if (!args.input || !args.output || !Number.isSafeInteger(args.limit) || args.limit < 1 ||
      !Number.isSafeInteger(args.seed) || !["broad", "priority_review"].includes(args.mode)) {
    throw new Error("Input/output paths and a positive integer limit/seed are required");
  }
  args.input = resolve(args.input);
  args.output = resolve(args.output);
  if (args["exclude-ids"]) args["exclude-ids"] = resolve(args["exclude-ids"]);
  return args;
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function goldKind(answer) {
  if (/^(?:[$£€])?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?$/.test(answer)) {
    return "numeric";
  }
  if (/^[A-Za-z][A-Za-z-]{0,19}$/.test(answer) &&
      !/^(?:yes|no|true|false|none|unknown|na|n-a|approximately|about)$/i.test(answer)) {
    return "single_word";
  }
  return null;
}

function normalizedQuestion(question) {
  return question.trim().toLowerCase().replace(/\s+/g, " ").replace(/[?.!]+$/g, "");
}

function riskFlags(question, answer) {
  return {
    gold_percent: /%/.test(answer),
    gold_currency: /[$£€]/.test(answer),
    gold_decimal_three_plus: /\.\d{3,}/.test(answer),
    gold_thousands_separator: /\d,\d{3}/.test(answer),
    question_asks_approximation: /\b(?:approximate(?:ly)?|estimated?|estimating|roughly|about|around|nearly|almost|nearest|round(?:ed|ing)?|close to)\b/i.test(question),
  };
}

function bareNumber(answer) {
  return /^[+-]?(?:0|[1-9]\d*)(?:\.\d{1,2})?$/.test(answer);
}

function candidate(row, lineNumber, seed, mode) {
  if (row.excluded_reason !== null && row.excluded_reason !== undefined && row.excluded_reason !== "") {
    return { reject: "source_excluded_reason" };
  }
  if (!["id", "question", "final_answer", "image"].every((key) => typeof row[key] === "string" && row[key].trim())) {
    return { reject: "missing_required_field" };
  }
  const id = row.id.trim();
  const question = row.question.trim();
  const answer = row.final_answer.trim();
  const image = row.image.trim();
  if (!/^images\/[A-Za-z0-9_-]+\.png$/.test(image) ||
      row.image_id == null || image.slice(7, -4) !== String(row.image_id)) {
    return { reject: "invalid_image_reference" };
  }
  const kind = goldKind(answer);
  if (!kind) return { reject: "gold_not_strictly_scorable" };
  if (question.length < 20) return { reject: "question_too_short" };
  const flags = riskFlags(question, answer);
  if (mode === "priority_review") {
    if (!bareNumber(answer)) return { reject: "priority_gold_not_bare_number" };
    if (flags.question_asks_approximation) return { reject: "priority_approximation_question" };
    if (row.used_in_sft1_this_question === true) return { reject: "priority_sft1_exact_question" };
    if (row.in_dpo_pool === true) return { reject: "priority_dpo_pool" };
  }
  const reasoningCue = /\b(?:compare|comparison|difference|how many|total|sum|average|mean|median|maximum|minimum|highest|lowest|tallest|shortest|increase|decrease|change|percent|percentage|ratio|trend|larger|smaller|greater|less|more)\b/i.test(question);
  const score = (kind === "numeric" ? 100 : 50) + (reasoningCue ? 10 : 0);
  return {
    id,
    question,
    final_answer: answer,
    image_relative_path: image,
    image_id: String(row.image_id),
    source: typeof row.source === "string" && row.source.trim() ? row.source.trim() : "ECD",
    source_field_present: typeof row.source === "string" && Boolean(row.source.trim()),
    source_dataset: "ECD",
    answer_type: kind,
    used_in_sft1: row.used_in_sft1 === true,
    used_in_sft1_this_question: row.used_in_sft1_this_question === true,
    in_dpo_pool: row.in_dpo_pool === true,
    risk_flags: flags,
    metadata_rank: score,
    metadata_reasoning_cue: reasoningCue,
    selection_status: "metadata_only_requires_image_gold_and_leakage_review",
    source_line: lineNumber,
    tie_break: digest(`${seed}:${id}`),
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const summaryPath = `${args.output}.summary.json`;
  if (existsSync(args.output) || existsSync(summaryPath)) {
    throw new Error("Output or summary already exists; choose new paths");
  }
  const counts = { input_rows: 0, eligible_rows: 0, selected_rows: 0, rejects: {} };
  const byChart = new Map();
  const seenIds = new Set();
  const excludedIds = new Set();
  let excludedIdsSha256 = null;
  if (args["exclude-ids"]) {
    const excludedBody = readFileSync(args["exclude-ids"], "utf8");
    excludedIdsSha256 = digest(excludedBody);
    for (const [i, line] of excludedBody.split(/\r?\n/).entries()) {
      if (!line.trim()) continue;
      let entry;
      try { entry = JSON.parse(line); } catch { throw new Error(`Malformed exclude-ids JSONL line ${i + 1}`); }
      if (typeof entry.id !== "string" || !entry.id.trim() || typeof entry.reason !== "string" || !entry.reason.trim()) {
        throw new Error(`Exclude-ids line ${i + 1} requires id and reason`);
      }
      excludedIds.add(entry.id.trim());
    }
  }
  const inputHash = createHash("sha256");
  const stream = createReadStream(args.input);
  stream.on("data", (chunk) => inputHash.update(chunk));
  const lines = createInterface({ input: stream, crlfDelay: Infinity });
  for await (const line of lines) {
    if (!line.trim()) continue;
    counts.input_rows += 1;
    let parsed;
    try { parsed = JSON.parse(line); } catch {
      throw new Error(`Malformed JSON on nonempty line ${counts.input_rows}`);
    }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error(`Non-object JSON on nonempty line ${counts.input_rows}`);
    }
    if (excludedIds.has(parsed.id)) {
      counts.rejects.known_visual_gold_mismatch = (counts.rejects.known_visual_gold_mismatch ?? 0) + 1;
      continue;
    }
    const row = candidate(parsed, counts.input_rows, args.seed, args.mode);
    if (row.reject) {
      counts.rejects[row.reject] = (counts.rejects[row.reject] ?? 0) + 1;
      continue;
    }
    if (seenIds.has(row.id)) {
      counts.rejects.duplicate_id = (counts.rejects.duplicate_id ?? 0) + 1;
      continue;
    }
    seenIds.add(row.id);
    counts.eligible_rows += 1;
    const old = byChart.get(row.image_id);
    if (!old || row.metadata_rank > old.metadata_rank ||
        (row.metadata_rank === old.metadata_rank && row.tie_break < old.tie_break)) {
      byChart.set(row.image_id, row);
    }
  }
  const inputSha256 = inputHash.digest("hex");
  const uniqueChartRows = [...byChart.values()].sort((a, b) =>
    b.metadata_rank - a.metadata_rank || a.tie_break.localeCompare(b.tie_break));
  const selected = [];
  const selectedQuestions = new Set();
  let duplicateQuestions = 0;
  for (const row of uniqueChartRows) {
    const questionKey = normalizedQuestion(row.question);
    if (selectedQuestions.has(questionKey)) {
      duplicateQuestions += 1;
      continue;
    }
    selectedQuestions.add(questionKey);
    selected.push(row);
    if (selected.length === args.limit) break;
  }
  counts.unique_eligible_charts = uniqueChartRows.length;
  counts.duplicate_questions_skipped = duplicateQuestions;
  counts.selected_rows = selected.length;
  counts.selected_numeric = selected.filter((r) => r.answer_type === "numeric").length;
  counts.selected_single_word = selected.length - counts.selected_numeric;
  counts.selected_used_in_sft1 = selected.filter((r) => r.used_in_sft1).length;
  counts.selected_used_in_sft1_this_question = selected.filter((r) => r.used_in_sft1_this_question).length;
  counts.selected_in_dpo_pool = selected.filter((r) => r.in_dpo_pool).length;
  counts.selected_reasoning_cue = selected.filter((r) => r.metadata_reasoning_cue).length;
  counts.selected_risk_flags = Object.fromEntries(
    Object.keys(riskFlags("", "")).map((flag) => [flag, selected.filter((r) => r.risk_flags[flag]).length]));
  counts.selected_source_field_present = selected.filter((r) => r.source_field_present).length;
  const outputRows = selected.map(({ tie_break, ...row }) => row);
  const body = outputRows.map((row) => JSON.stringify(row)).join("\n") + "\n";
  await mkdir(dirname(args.output), { recursive: true });
  const writer = createWriteStream(args.output, { flags: "wx" });
  await new Promise((done, fail) => {
    writer.on("error", fail);
    writer.on("finish", done);
    writer.end(body);
  });
  const summary = {
    status: "metadata_only_not_approved_for_training",
    input: args.input,
    input_sha256: inputSha256,
    output: args.output,
    output_sha256: digest(body),
    selector_sha256: digest(readFileSync(new URL(import.meta.url))),
    mode: args.mode,
    excluded_ids_file: args["exclude-ids"] ?? null,
    excluded_ids_sha256: excludedIdsSha256,
    source_note: "A's source JSONL has no source column; ECD is the declared input dataset, not a per-item source field.",
    seed: args.seed,
    limit: args.limit,
    ...counts,
    required_next_checks: ["image_exists_and_readable", "gold_matches_image", "v1_2_reward_parity", "MMMU_and_MMMU_Pro_leakage", "C_approval"],
  };
  const summaryWriter = createWriteStream(summaryPath, { flags: "wx" });
  await new Promise((done, fail) => {
    summaryWriter.on("error", fail);
    summaryWriter.on("finish", done);
    summaryWriter.end(JSON.stringify(summary, null, 2) + "\n");
  });
  console.log(JSON.stringify(summary, null, 2));
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
