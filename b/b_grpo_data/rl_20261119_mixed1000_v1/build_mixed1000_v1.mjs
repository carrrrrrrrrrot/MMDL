#!/usr/bin/env node
// Rebuild a 500 ECD + 500 CLEVR exploratory training set from pinned B inputs.
// Selected JSONL rows are copied byte-for-byte (apart from line ordering).

import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const defaults = {
  ecd: resolve(here, '../rl_20261119/ecd_qc_v2/ecd_manifest.jsonl'),
  clevr: resolve(here, '../rl_20261119_clevr_v3/mammoth_clevr_manifest.jsonl'),
};
const expected = {
  ecd: 'c05d5962a4b028236c726c72246eb9e8378d2ee0d96c9db3d0912f108625f1ed',
  clevr: 'b2edb0e30acbdca0a0d64800405790fb5926f92af3f73c5e47072f3ff570a33d',
};
const tag = 'mmdl-b-mixed1000-v1';
// Keep the heldout IDs identical to mixed_v2 for a paired internal check.
const heldoutTag = 'mmdl-b-mixed-v2';
const seed = 3407;

function sha(data) {
  return createHash('sha256').update(data).digest('hex');
}

function fail(message) {
  throw new Error(message);
}

function argsFrom(argv) {
  const args = { ecd: defaults.ecd, clevr: defaults.clevr };
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!['--ecd', '--clevr', '--output'].includes(key) || !value) {
      fail('Usage: node build_mixed1000_v1.mjs [--ecd PATH] [--clevr PATH] --output NEW_JSONL');
    }
    args[key.slice(2)] = resolve(value);
  }
  if (!args.output) fail('--output is required');
  return args;
}

function readPinned(path, label) {
  const data = readFileSync(path);
  const actual = sha(data);
  if (actual !== expected[label]) fail(`${label}: SHA256 ${actual} != ${expected[label]}`);
  const lines = data.toString('utf8').split(/\r?\n/).filter(Boolean);
  const ids = new Set();
  const rows = lines.map((line, index) => {
    const row = JSON.parse(line);
    if (!['train', 'heldout'].includes(row.split)) fail(`${label}:${index + 1}: invalid split`);
    if (!row.sample_id || ids.has(row.sample_id)) fail(`${label}:${index + 1}: duplicate/empty sample_id`);
    if (!row.original_id || !row.question || !row.answer || !row.answer_type) {
      fail(`${label}:${index + 1}: missing required QA field`);
    }
    if (!Array.isArray(row.image_paths) || row.image_paths.length !== 1 ||
        !/^[0-9a-f]{64}$/.test(row.image_sha256)) {
      fail(`${label}:${index + 1}: invalid image reference/hash`);
    }
    if (!Array.isArray(row.prompt) || row.prompt[0]?.role !== 'system' ||
        row.prompt[1]?.role !== 'user') {
      fail(`${label}:${index + 1}: invalid prompt structure`);
    }
    ids.add(row.sample_id);
    return { line, row, label };
  });
  return { rows, sha256: actual };
}

function choose(rows, label, split, count, rankTag = tag) {
  const pool = rows.filter(({ row }) => row.split === split);
  if (pool.length < count) fail(`${label}: ${split} has ${pool.length}, needs ${count}`);
  return pool
    .map((item) => ({ ...item, rank: sha(`${rankTag}\0${seed}\0${label}\0${split}\0${item.row.sample_id}`) }))
    .sort((a, b) => a.rank.localeCompare(b.rank) || a.row.sample_id.localeCompare(b.row.sample_id))
    .slice(0, count);
}

function count(rows, predicate) {
  return rows.filter(({ row }) => predicate(row)).length;
}

function main() {
  const args = argsFrom(process.argv.slice(2));
  const reportPath = resolve(dirname(args.output), 'selection_report.json');
  if (existsSync(args.output) || existsSync(reportPath)) {
    fail('Output manifest or report already exists; refuse to overwrite');
  }
  if (!existsSync(dirname(args.output))) fail('Output directory must already exist');

  const ecd = readPinned(args.ecd, 'ecd');
  const clevr = readPinned(args.clevr, 'clevr');
  const train = [
    ...choose(ecd.rows, 'ecd', 'train', 500),
    ...choose(clevr.rows, 'clevr', 'train', 500),
  ];
  const heldout = [
    ...choose(ecd.rows, 'ecd', 'heldout', 50, heldoutTag),
    ...choose(clevr.rows, 'clevr', 'heldout', 50, heldoutTag),
  ];
  for (const group of [train, heldout]) {
    group.sort((a, b) =>
      sha(`${tag}-order\0${seed}\0${a.row.sample_id}`).localeCompare(
        sha(`${tag}-order\0${seed}\0${b.row.sample_id}`)) ||
      a.row.sample_id.localeCompare(b.row.sample_id));
  }
  const combined = [...train, ...heldout];
  for (const [field, value] of [
    ['sample_id', (x) => x.sample_id],
    ['original_id', (x) => x.original_id],
    ['image_sha256', (x) => x.image_sha256],
    ['image_path', (x) => x.image_paths[0]],
  ]) {
    const seen = new Set();
    for (const { row } of combined) {
      const key = value(row);
      if (!key || seen.has(key)) fail(`Selected ${field} is empty or reused: ${key}`);
      seen.add(key);
    }
  }
  if (train.length !== 1000 || heldout.length !== 100) fail('Wrong mixed split size');
  for (const item of train) if (item.row.split !== 'train') fail('Train/heldout leakage');
  for (const item of heldout) if (item.row.split !== 'heldout') fail('Train/heldout leakage');

  const output = Buffer.from(combined.map(({ line }) => line).join('\n') + '\n', 'utf8');
  const report = {
    schema: 'b_rl_mixed_500_500_selection_v1',
    status: 'exploratory_not_gold_verified_or_C_training_approved',
    seed,
    hash_tag: tag,
    heldout_hash_tag: heldoutTag,
    source_sha256: { ecd: ecd.sha256, clevr: clevr.sha256 },
    source_counts: {
      ecd: { train: count(ecd.rows, (r) => r.split === 'train'), heldout: count(ecd.rows, (r) => r.split === 'heldout') },
      clevr: { train: count(clevr.rows, (r) => r.split === 'train'), heldout: count(clevr.rows, (r) => r.split === 'heldout') },
    },
    selected: { train: 1000, heldout: 100, ecd_train: 500, clevr_train: 500, ecd_heldout: 50, clevr_heldout: 50 },
    selected_train_clevr_a_overlap: {
      sft1: count(train.filter((x) => x.label === 'clevr'), (r) => (r.a_used_in || []).includes('sft1')),
      dpo_pool_or_pair: count(train.filter((x) => x.label === 'clevr'), (r) =>
        (r.a_used_in || []).includes('dpo_pool') || (r.a_used_in || []).includes('dpo_pairs')),
    },
    selected_heldout_clevr_base_k4: count(heldout.filter((x) => x.label === 'clevr'),
      (r) => r.a_base_k_correct === 4),
    unique_sample_ids: 1100,
    unique_original_ids: 1100,
    unique_image_hashes: 1100,
    unique_image_paths: 1100,
    output_sha256: sha(output),
    note: 'Rows copied from pinned inputs unchanged; source image/gold quality is not re-reviewed by this builder.',
  };
  writeFileSync(args.output, output, { flag: 'wx' });
  writeFileSync(reportPath, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify(report, null, 2));
}

main();
