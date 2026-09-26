"""보조 분석: 규칙 파서가 답 글자를 찾지 못한 객관식 응답만 LLM judge 로 보기 글자에 매핑해 봅니다.

본 평가 기준(엄격 채점, 평가 기준 v1.1)은 바꾸지 않습니다. 이미 저장된 samples.jsonl 을 읽어 "judge 보조 점수"를 따로 계산합니다.

  대상   : mmmu_acc.random_fallback_pred 가 기록된 객관식 문항 (= 공식 파서가 후보를 찾지 못해 엄격 채점에서 오답 처리된 문항)
  judge  : 문제 본문과 정답은 넣지 않고, 보기 목록과 모델 응답만 넣습니다 (judge 가 문제를 대신 풀지 못하게 하기 위함)
           응답에 최종적으로 고른 보기가 없으면 INVALID → 오답 유지
  결과   : judge 호출 결과를 캐시 파일(jsonl)에 저장합니다. 같은 캐시로 다시 계산하면 결과가 같습니다.
           (API 모델은 temperature=0 이어도 완전히 결정적이지 않으므로, 재현은 캐시로 보장합니다)

사용:
  # 호출 수와 입력 크기만 확인 (API 호출 없음)
  python scripts/judge/llm_judge_rescore.py results/runs/M4_final --dry_run
  # 실제 실행 (OPENAI_API_KEY 환경 변수 필요)
  python scripts/judge/llm_judge_rescore.py results/runs/M4_final --model <모델명> --cache results/judge/M4_final_<모델명>.jsonl
"""

import argparse
import ast
import glob
import json
import os
import sys

TAIL_CHARS = 6000  # 응답이 길면(최대 16384 토큰) 마지막 부분만 보냅니다. 결론은 보통 끝에 있습니다.

PROMPT = """You are an answer-format parser, not a problem solver.
Map the MODEL RESPONSE to exactly one of the provided options, but only if the response itself commits to a final choice.
Do not solve the question yourself. Do not guess the correct answer.
If the response never commits to a final choice (for example it is cut off, keeps reconsidering, or only discusses options), return INVALID.

OPTIONS:
{options}

MODEL RESPONSE (possibly truncated at the beginning):
{response}

Return exactly one valid option letter or INVALID."""


def load_docs_options(data_root):
    import pyarrow.parquet as pq

    opts = {}
    for path in sorted(glob.glob(os.path.join(data_root, "*", "validation-*.parquet"))):
        for doc in pq.read_table(path, columns=["id", "options"]).to_pylist():
            opts[doc["id"]] = ast.literal_eval(doc["options"])
    return opts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--data_root", default="/mnt/SSD-2TB/models/MMMU")
    ap.add_argument("--model")
    ap.add_argument("--cache")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    options = load_docs_options(args.data_root)
    targets = []
    for seed_dir in sorted(glob.glob(os.path.join(args.run_dir, "seed_*"))):
        seed = os.path.basename(seed_dir).removeprefix("seed_")
        path = glob.glob(os.path.join(seed_dir, "*", "*_samples_*.jsonl"))[0]
        for r in map(json.loads, open(path)):
            acc = r["mmmu_acc"]
            if acc["question_type"] == "multiple-choice" and acc.get("random_fallback_pred") is not None:
                targets.append((seed, acc["id"], acc["answer"], r["filtered_resps"][-TAIL_CHARS:]))

    total_chars = sum(len(t[3]) for t in targets)
    print(f"대상 {len(targets)}건 (seed별: " + ", ".join(f"{s} {sum(t[0] == s for t in targets)}" for s in sorted({t[0] for t in targets})) + ")")
    print(f"보낼 응답 텍스트 합계 {total_chars:,}자 (대략 {total_chars // 4:,} 토큰)")
    if args.dry_run:
        return
    if not (args.model and args.cache):
        sys.exit("--model 과 --cache 가 필요합니다.")

    cache = {}
    if os.path.exists(args.cache):
        for line in open(args.cache):
            c = json.loads(line)
            cache[(c["seed"], c["id"])] = c

    from openai import OpenAI

    client = OpenAI()  # OPENAI_API_KEY 환경 변수를 읽습니다
    os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
    with open(args.cache, "a") as f:
        for seed, doc_id, _gold, resp in targets:
            if (seed, doc_id) in cache:
                continue
            letters = [chr(65 + j) for j in range(len(options[doc_id]))]
            prompt = PROMPT.format(options="\n".join(f"{l}. {o}" for l, o in zip(letters, options[doc_id])), response=resp)
            out = client.responses.create(model=args.model, input=prompt, temperature=0)
            raw = out.output_text.strip().upper()
            pred = raw if raw in letters else None
            rec = {"seed": seed, "id": doc_id, "model": args.model, "raw": raw, "pred": pred}
            cache[(seed, doc_id)] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\njudge 모델: {args.model}, 캐시: {args.cache}")
    for seed in sorted({t[0] for t in targets}):
        ts = [t for t in targets if t[0] == seed]
        mapped = [t for t in ts if cache[(seed, t[1])]["pred"] is not None]
        hit = sum(cache[(seed, t[1])]["pred"] == t[2] for t in ts)
        print(f"seed {seed}: 대상 {len(ts)} | judge 가 글자로 매핑 {len(mapped)} | 그중 정답 {hit} (엄격 점수에 더하면 +{hit / 9:.2f}pt)")


if __name__ == "__main__":
    main()
