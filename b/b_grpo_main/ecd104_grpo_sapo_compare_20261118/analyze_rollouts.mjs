/** Audit repeated TRL completion logs when num_iterations=2. */
import { readFileSync, writeFileSync } from "node:fs";

if (process.argv.length !== 5) {
  throw new Error("Usage: node analyze_rollouts.mjs GRPO.jsonl SAPO.jsonl OUTPUT.json");
}
const [, , grpoPath, sapoPath, outputPath] = process.argv;
const read = (path) => readFileSync(path, "utf8").trim().split(/\r?\n/).map(JSON.parse);
const grpo = read(grpoPath);
const sapo = read(sapoPath);

function audit(rows) {
  const steps = new Map();
  for (const row of rows) {
    if (!steps.has(row.step)) steps.set(row.step, []);
    steps.get(row.step).push(row);
  }
  const stepNumbers = [...steps.keys()].sort((a, b) => a - b);
  if (stepNumbers.length !== 64 || stepNumbers.some((step, index) => step !== index + 1)) {
    throw new Error("Expected optimizer steps 1..64");
  }
  let repeatedPairsIdentical = 0;
  const generated = [];
  for (let step = 1; step <= 64; step += 2) {
    const first = steps.get(step).sort((a, b) => a.candidate_index - b.candidate_index);
    const second = steps.get(step + 1).sort((a, b) => a.candidate_index - b.candidate_index);
    if (first.length !== 4 || second.length !== 4) throw new Error(`Wrong rollout count at ${step}`);
    if (first.every((row, i) =>
      row.sample_id === second[i].sample_id &&
      row.raw_output === second[i].raw_output &&
      row.reward === second[i].reward)) {
      repeatedPairsIdentical += 1;
    }
    generated.push(...first);
  }
  const groups = Array.from({ length: 32 }, (_, i) => generated.slice(4 * i, 4 * i + 4));
  return {
    logged_rows: rows.length,
    logged_optimizer_groups: steps.size,
    generated_groups: groups.length,
    generated_rows: generated.length,
    repeated_optimizer_step_pairs_identical: repeatedPairsIdentical,
    mixed_reward_groups: groups.filter((group) => new Set(group.map((row) => row.reward)).size > 1).length,
    mean_reward: generated.reduce((sum, row) => sum + row.reward, 0) / generated.length,
    reward_one_count: generated.filter((row) => row.reward === 1).length,
    missing_final_answer_count: generated.filter((row) => row.reason === "missing_final_answer").length,
    clipped_proxy_count: generated.filter((row) => row.clipped_proxy === true).length,
    generated,
  };
}

const a = audit(grpo);
const b = audit(sapo);
let sameId = 0;
let sameOutput = 0;
let sameReward = 0;
let grpoOnlyCorrect = 0;
let sapoOnlyCorrect = 0;
for (let i = 0; i < a.generated.length; i += 1) {
  const left = a.generated[i];
  const right = b.generated[i];
  sameId += Number(left.sample_id === right.sample_id && left.candidate_index === right.candidate_index);
  sameOutput += Number(left.raw_output === right.raw_output);
  sameReward += Number(left.reward === right.reward);
  grpoOnlyCorrect += Number(left.reward === 1 && right.reward === 0);
  sapoOnlyCorrect += Number(left.reward === 0 && right.reward === 1);
}
const { generated: _left, ...grpoSummary } = a;
const { generated: _right, ...sapoSummary } = b;
const result = {
  status: "audit_of_logged_training_rollouts_not_benchmark_evaluation",
  num_iterations: 2,
  grpo: grpoSummary,
  sapo: sapoSummary,
  paired_generated_rows: a.generated.length,
  same_sample_and_candidate: sameId,
  identical_raw_outputs: sameOutput,
  identical_binary_rewards: sameReward,
  grpo_only_reward_one: grpoOnlyCorrect,
  sapo_only_reward_one: sapoOnlyCorrect,
};
writeFileSync(outputPath, JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify(result, null, 2));
