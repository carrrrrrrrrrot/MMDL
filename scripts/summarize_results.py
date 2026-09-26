"""run_mmmu_eval.sh 결과 폴더(seed 여러 개)를 읽어 과목별 표와 요약 지표를 만듭니다.

채점은 run 이 samples.jsonl 에 남긴 parsed_pred 를 lmms-eval 의 MMMU 공식 판정 함수
(객관식: 글자 일치, open: eval_open)로 다시 판정합니다.
  1) 기록된 정답 그대로 판정한 합계가 lmms-eval 의 mmmu_acc 와 같은지 먼저 확인하고 (기록 정합성),
  2) 현재 평가 기준(v1.2)으로 다시 판정합니다: 정답 목록 정규화(normalize_gold, v1.1),
     open 최종 답의 LaTeX 정규화(normalize_latex, v1.2). 이전 버전 run(M4_final 등)은 이 단계에서 재채점됩니다.

  종합 점수(엄격) = 30과목 정확도의 단순 평균 = 900문항 중 정답 수 / 900 (과목당 30문항으로 같으므로 동일)
  공식 관례 점수   = 엄격 점수 + 무응답 객관식에 공식 파서의 random fallback 을 적용했을 때 맞는 문항

사용: python scripts/summarize_results.py results/runs/M4_final [--out results/runs/M4_final/summary.md]
"""

import argparse
import glob
import importlib.util
import json
import os
import statistics as st

from lmms_eval.tasks.mmmu.utils import eval_open, parse_open_response

_spec = importlib.util.spec_from_file_location(
    "mmdl_utils", os.path.join(os.path.dirname(__file__), "..", "eval_tasks", "mmmu_mmdl", "mmdl_utils.py")
)
mmdl_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mmdl_utils)


def is_correct(acc, normalize=True):
    """normalize=False: run 이 기록한 parsed_pred 와 정답 그대로 판정 (lmms-eval 기록 재현).
    normalize=True : 현재 평가 기준(v1.2)으로 재판정. open 문항은 정답 목록 정규화(v1.1)와
                     최종 답의 LaTeX 정규화 후 공식 parse_open_response 로 다시 파싱(v1.2)합니다."""
    pred = acc["parsed_pred"][0]
    if acc["question_type"] == "multiple-choice":
        return pred == acc["answer"]
    if not normalize:
        return eval_open(acc["answer"], pred)
    if acc.get("final_answer") is not None:
        pred = parse_open_response(mmdl_utils.normalize_latex(acc["final_answer"]))
    return eval_open(mmdl_utils.normalize_gold(acc), pred)


def load_seed(seed_dir):
    samples = glob.glob(os.path.join(seed_dir, "*", "*_samples_*.jsonl"))[0]
    results = glob.glob(os.path.join(seed_dir, "*", "*_results.json"))[0]
    with open(results) as f:
        res = json.load(f)
    task = next(iter(res["results"]))
    with open(samples) as f:
        rows = [json.loads(line)["mmmu_acc"] for line in f]
    return rows, res["results"][task]["mmmu_acc,none"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out")
    args = ap.parse_args()

    seed_dirs = sorted(glob.glob(os.path.join(args.run_dir, "seed_*")))
    seeds = [os.path.basename(d).removeprefix("seed_") for d in seed_dirs]
    per_seed = {}
    for s, d in zip(seeds, seed_dirs):
        rows, lmms_acc = load_seed(d)
        assert len(rows) == 900, f"{d}: 문항 수 {len(rows)}"
        recorded = [is_correct(r, normalize=False) for r in rows]
        assert abs(sum(recorded) / 900 - lmms_acc) < 1e-4, f"{d}: 기록 기준 재계산 {sum(recorded) / 900} != lmms-eval {lmms_acc}"
        correct = [is_correct(r) for r in rows]
        fallback_hit = sum(r.get("random_fallback_pred") is not None and r["random_fallback_pred"] == r["answer"] for r in rows)
        per_seed[s] = {"rows": rows, "correct": correct, "recorded": sum(recorded), "fallback_hit": fallback_hit}

    subjects = sorted({r["subdomain"] for r in per_seed[seeds[0]]["rows"]})
    lines = [f"# MMMU val 결과: {args.run_dir}", "", f"seed: {', '.join(seeds)}", ""]
    lines += ["| No. | Subject | Data Num | " + " | ".join(f"seed {s}" for s in seeds) + " | Acc (mean ± std) |", "|---|---|---|" + "---|" * len(seeds) + "---|"]
    for i, subj in enumerate(subjects, 1):
        accs = []
        for s in seeds:
            idx = [j for j, r in enumerate(per_seed[s]["rows"]) if r["subdomain"] == subj]
            accs.append(100 * sum(per_seed[s]["correct"][j] for j in idx) / len(idx))
        sd = st.stdev(accs) if len(accs) > 1 else 0.0
        lines.append(f"| {i} | {subj} | {len(idx)} | " + " | ".join(f"{a:.1f}" for a in accs) + f" | {st.mean(accs):.1f} ± {sd:.1f} |")

    overall = [100 * sum(per_seed[s]["correct"]) / 900 for s in seeds]
    official = [100 * (sum(per_seed[s]["correct"]) + per_seed[s]["fallback_hit"]) / 900 for s in seeds]
    sd = lambda xs: st.stdev(xs) if len(xs) > 1 else 0.0  # noqa: E731
    lines.append(
        "| | **Overall (macro avg)** | **900** | " + " | ".join(f"**{a:.2f}**" for a in overall) + f" | **{st.mean(overall):.2f} ± {sd(overall):.2f}** |"
    )
    lines += ["", "계산식: Overall = mean(30개 과목 accuracy) = 정답 수 / 900 (과목당 30문항). mean ± std 는 seed 간 표본 표준편차.", ""]

    lines += ["| 지표 | " + " | ".join(f"seed {s}" for s in seeds) + " |", "|---|" + "---|" * len(seeds)]

    def row(name, fn):
        lines.append(f"| {name} | " + " | ".join(str(fn(per_seed[s])) for s in seeds) + " |")

    row("정답 수 (엄격, 평가 기준 v1.2)", lambda p: sum(p["correct"]))
    row("참고: run 당시 lmms-eval 기록 정답 수", lambda p: p["recorded"])
    row("객관식 정답 / 847", lambda p: sum(c for r, c in zip(p["rows"], p["correct"]) if r["question_type"] == "multiple-choice"))
    row("open 정답 / 53", lambda p: sum(c for r, c in zip(p["rows"], p["correct"]) if r["question_type"] != "multiple-choice"))
    row("`Answer:` 없음 (final_answer=None)", lambda p: sum(r.get("final_answer") is None for r in p["rows"]))
    row("객관식 답 글자 없음 (엄격 오답 처리)", lambda p: sum(r.get("random_fallback_pred") is not None for r in p["rows"]))
    row("공식 관례 점수 (%)", lambda p: f"{100 * (sum(p['correct']) + p['fallback_hit']) / 900:.2f}")
    lines += ["", f"공식 관례 점수 평균: {st.mean(official):.2f}", ""]

    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
