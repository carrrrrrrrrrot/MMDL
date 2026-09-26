# validation_Math_5

- 과목: Math / 유형: multiple-choice
- **정답: A**
- 모델 최종 답 (seed 3407 / 1 / 2): C / C / A
- seed 3407 추출 줄: `C`

## 이미지

**image 1** (584×586 px)

![image 1](../images/validation_Math_5_1.png)

## 문제

<image 1>The region bounded by the graph as shown above.Choose an integral expression that can be used to find the area of R

- **A.** $\int_{0}^{1.5}[f(x)-g(x)]\,d x$
- **B.** $\int_{0}^{1.5}[g(x)-f(x)]\,d x$
- **C.** $\int_{0}^{2}[f(x)-g(x)]\,d x$
- **D.** $\int_{0}^{2}[g(x)-f(x)]\,d x$

## 모델 응답 (seed 3407 전문)

```text
We are given two functions:

- $ f(x) = x^3 - 6x^2 + 8x $
- $ g(x) = -\frac{1}{2}x^2 + 2x $

And a graph showing two regions: R and S, bounded between the curves and the x-axis.

We are to find an integral expression for the area of region **R**.

---

### Step 1: Understand the region R

Looking at the graph:

- Region R is the shaded area **above** the curve g(x) and **below** the curve f(x), from x=0 to x=2.
- At x=0, both curves pass through the origin (O).
- At x=2, they intersect (marked as (2,0)) — this is the right boundary of region R.

So, in the interval [0, 2], f(x) is above g(x). We can verify this by plugging in a value, say x=1:

- f(1) = 1 - 6 + 8 = 3
- g(1) = -0.5 + 2 = 1.5
→ So f(1) > g(1), meaning f(x) is above g(x) on [0,2].

Therefore, the area of R is:

$$
\int_{0}^{2} [f(x) - g(x)] \, dx
$$

---

### Step 2: Check the options

A. $\int_{0}^{1.5}[f(x)-g(x)]\,d x$ → This is only up to 1.5, but R goes to x=2. ❌

B. $\int_{0}^{1.5}[g(x)-f(x)]\,d x$ → Negative sign and wrong limits. ❌

C. $\int_{0}^{2}[f(x)-g(x)]\,d x$ → Matches our reasoning. ✅

D. $\int_{0}^{2}[g(x)-f(x)]\,d x$ → This would be negative of the correct area. ❌

---

### Final Answer:

Answer: C
```
