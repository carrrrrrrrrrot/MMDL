# validation_Math_15

- 과목: Math / 유형: open
- **정답: ['24/7', '3.429']**
- 모델 최종 답 (seed 3407 / 1 / 2): 24/7 / 24/7 / $\frac{24}{7}$
- seed 3407 추출 줄: `24/7`

## 이미지

**image 1** (489×267 px)

![image 1](../images/validation_Math_15_1.png)

## 문제

The top of a 25-foot ladder, leaning against a vertical wall is slipping down the wall at the rate of 1 foot per second. How fast is the bottom of the ladder slipping along the ground when the bottom of the ladder is 7 feet away from the base of the wall? <image 1>



## 모델 응답 (seed 3407 전문)

```text
We are given a right triangle formed by a ladder leaning against a wall, with the ladder as the hypotenuse (length 25 feet), the ground as one leg (length x), and the wall as the other leg (length y).

We are told:
- The top of the ladder (y) is slipping down the wall at a rate of dy/dt = -1 ft/s (negative because it’s decreasing).
- We need to find dx/dt, the rate at which the bottom of the ladder is slipping along the ground, when x = 7 ft.

We use the Pythagorean theorem:

x² + y² = 25² = 625

Differentiate both sides with respect to time t:

2x(dx/dt) + 2y(dy/dt) = 0

Divide both sides by 2:

x(dx/dt) + y(dy/dt) = 0

Now solve for dx/dt:

dx/dt = - [y / x] * (dy/dt)

We are given x = 7 ft. We need to find y when x = 7.

From the Pythagorean theorem:

y² = 625 - x² = 625 - 49 = 576

So y = √576 = 24 ft

Now plug into the equation:

dx/dt = - [24 / 7] * (-1) = (24 / 7) ft/s

The negative signs cancel, so the bottom is moving away from the wall at 24/7 ft/s.

Answer: 24/7
```
