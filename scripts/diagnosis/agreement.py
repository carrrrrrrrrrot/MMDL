"""labels.csv 의 claude_label 과 user_label 의 일치도를 계산합니다.

사용: python scripts/diagnosis/agreement.py results/diagnosis_eye_brain/labels.csv
"""

import collections
import csv
import sys

rows = [r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r["user_label"].strip()]
if not rows:
    sys.exit("user_label 이 채워진 행이 없습니다.")

a = [r["claude_label"].strip() for r in rows]
b = [r["user_label"].strip() for r in rows]
n = len(rows)
agree = sum(x == y for x, y in zip(a, b))

# Cohen's kappa
labels = sorted(set(a) | set(b))
pa, pb = collections.Counter(a), collections.Counter(b)
pe = sum(pa[k] * pb[k] for k in labels) / (n * n)
po = agree / n
kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

print(f"판정한 문항: {n}")
print(f"일치: {agree}/{n} = {po:.1%}")
print(f"Cohen's kappa: {kappa:.3f}")
print()
print("혼동표 (행 = Claude, 열 = 사용자)")
print("\t" + "\t".join(labels))
for x in labels:
    print(x + "\t" + "\t".join(str(sum(1 for i, j in zip(a, b) if i == x and j == y)) for y in labels))
print()
for r, x, y in zip(rows, a, b):
    if x != y:
        print(f"불일치 {r['id']}: Claude={x} / 사용자={y}")
        print(f"  Claude 근거: {r['claude_evidence']}")
        print(f"  사용자 근거: {r['user_note']}")
