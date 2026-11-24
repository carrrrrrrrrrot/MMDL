"""MMMU test run 요약: 2026-02 공개 정답으로 로컬 채점 (v1.2 is_correct: 객관식 엄격, open 정규화), 무응답, 과목별, 제출 JSON 점검.
실행: ~/anaconda3/envs/vllm011/bin/python -I local/eval_mmmu_test/test_summary.py RUN_SEED_DIR
"""
import collections, glob, hashlib, json, os, sys

sys.path.insert(0, "/home/dekim/workspace/MMDL/scripts")
import summarize_results as S  # noqa: E402

d = sys.argv[1]
rows, lmms_acc = S.load_seed(d)
sub_path = os.path.join(d, "submissions", "mmmu_test_for_submission.json")
sub = json.load(open(sub_path))
per = collections.defaultdict(lambda: [0, 0, 0])  # n, correct, no_answer
for r in rows:
    p = per[r["subdomain"]]; p[0] += 1; p[1] += int(S.is_correct(r)); p[2] += r.get("final_answer") is None
n = sum(v[0] for v in per.values()); c = sum(v[1] for v in per.values()); na = sum(v[2] for v in per.values())
mc_fb = sum(1 for r in rows if r["question_type"] == "multiple-choice" and r.get("random_fallback_pred") is not None)
open_empty = sum(1 for r in rows if r["question_type"] != "multiple-choice" and sub.get(r["id"], "") == "")
out = {"scoring": "2026-02 공개 정답으로 로컬 채점 (v1.2 is_correct)", "n": n, "correct": c, "acc_pct": round(100 * c / n, 2),
       "no_answer": na, "mc_random_fallback_in_submission": mc_fb, "open_empty_in_submission": open_empty,
       "lmms_eval_mmmu_acc": lmms_acc,
       "submission": {"path": sub_path, "sha256": hashlib.sha256(open(sub_path, "rb").read()).hexdigest(), "n_ids": len(sub),
                      "ids_match_rows": set(sub) == {r["id"] for r in rows}},
       "per_subject": {k: {"n": v[0], "correct": v[1], "acc_pct": round(100 * v[1] / v[0], 1), "no_answer": v[2]} for k, v in sorted(per.items())}}
print(json.dumps(out, ensure_ascii=False, indent=1))
