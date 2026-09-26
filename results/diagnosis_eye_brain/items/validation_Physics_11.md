# validation_Physics_11

- 과목: Physics / 유형: multiple-choice
- **정답: B**
- 모델 최종 답 (seed 3407 / 1 / 2): C / B / D
- seed 3407 추출 줄: `C`

## 이미지

**image 1** (524×282 px)

![image 1](../images/validation_Physics_11_1.png)

## 문제

<image 1>A light ray is incident normal to a thin layer of glass. Given the figure, what is the minimum thickness of the glass that gives the reflected light an orange like color ($\lambda $(air) orange light = 600nm)

- **A.** 50 nm
- **B.** 100 nm
- **C.** 150 nm
- **D.** 200 nm

## 모델 응답 (seed 3407 전문)

```text
We are given a thin film of glass (n = 1.50) sandwiched between air (n = 1.00) above and water (n = 1.33) below. A light ray is incident **normal** to the glass layer. We are to find the **minimum thickness t** of the glass such that the reflected light has an orange color, corresponding to λ = 600 nm in air.

---

### Step 1: Understand the interference

Since the ray is incident normally, we are dealing with **normal-incidence thin-film interference**. The key is to determine whether the phase change occurs at each interface.

---

### Step 2: Phase changes upon reflection

- At the **air-glass interface**: light reflects from a medium of higher index (air n=1.00 → glass n=1.50). So, **phase change of π radians (180°)** occurs.
- At the **glass-water interface**: light reflects from a medium of higher index (glass n=1.50 → water n=1.33). Since 1.50 > 1.33, **phase change of π radians (180°)** also occurs.

So, both reflections involve a phase change of π. That means the **net phase difference between the two reflected rays is 0** — they are **in phase**.

Wait — let’s double-check:

Actually, when light reflects off a boundary where it goes from a lower to higher refractive index, there is a 180° phase shift. When it reflects from higher to lower, there is no phase shift.

So:

- Air to glass: n_air < n_glass → phase shift of π
- Glass to water: n_glass > n_water → NO phase shift

Therefore, the two reflected rays have **one phase shift** (from air-glass) and **no phase shift** (from glass-water), so the **total phase difference is π** — meaning they are **out of phase by 180°**.

That is important.

---

### Step 3: Path difference and condition for destructive or constructive interference

The path difference between the two rays is due to the round trip through the glass: 2t.

But since there is a phase shift of π (180°) due to reflection at the first interface, and no phase shift at the second, the two reflected rays are out of phase by π.

Therefore, for **constructive interference** (bright reflection), we need the path difference to be an **odd multiple of half-wavelengths**, because the waves are already out of phase by π.

In other words, the condition for **constructive interference** (bright reflection) is:

> 2t = (m + ½)λ  
> where m = 0, 1, 2, ... (for minimum thickness, take m=0)

So,

> t = (m + ½) * λ / 2

For minimum thickness, m=0:

> t = (½) * λ / 2 = λ / 4

Wait — let me re-express:

From above:

Path difference = 2t

Phase difference = π (due to reflection) + (2π/λ) * 2t

Total phase difference = π + (4πt)/λ

For constructive interference, total phase difference should be 2πk, for integer k.

So:

π + (4πt)/λ = 2πk

Divide both sides by π:

1 + (4t)/λ = 2k

→ (4t)/λ = 2k - 1

→ t = [ (2k - 1) * λ ] / 4

So for minimum thickness, k=1:

t = (2*1 - 1) * λ / 4 = λ / 4

Wait — this contradicts what I had before. Let's check again.

Actually, let’s use the standard formula for normal incidence on a thin film.

---

### Standard formula for thin film interference:

When light reflects off a thin film, the condition for **destructive interference** (darkness) in reflection depends on the phase shifts.

Case 1: If both reflections have phase shifts → net phase difference = 0 → constructive interference when path difference = mλ

Case 2: If only one reflection has phase shift → net phase difference = π → constructive interference when path difference = (m + ½)λ

In our case:

- Reflection at air-glass: phase shift
- Reflection at glass-water: no phase shift

So net phase difference = π

Thus, for **constructive interference** (bright reflection):

path difference = (m + ½)λ

But path difference = 2t

So:

2t = (m + ½)λ

→ t = (m + ½)λ / 2

Minimum thickness → m=0:

t = (½)λ / 2 = λ/4

Wait — that gives t = λ/4 = 600 nm / 4 = 150 nm

But let’s confirm this.

Actually, let’s think about the wavelength inside the film.

Is the wavelength in air or in glass?

The problem says “λ (air) orange light = 600 nm”

In thin film interference problems, unless specified otherwise, we usually use the wavelength in vacuum or air for calculations — but we must be careful.

Actually, the condition for constructive interference in reflection is often written using the **wavelength in the medium**.

But in many textbooks, especially for normal incidence, they use the **wavelength in air** for the condition, assuming the phase shifts are accounted for.

Let me derive it properly.

---

### Correct derivation:

Let’s denote:

- Incident ray: from air to glass
- Ray 1: reflects off top surface (air-glass)
- Ray 2: enters glass, reflects off bottom (glass-water), then exits back to air

Phase shift at air-glass: π

Phase shift at glass-water: 0 (since n_glass > n_water)

So the two reflected rays are out of phase by π.

Now, the path difference between them is 2t.

The wave travels through the glass and back, so the extra distance traveled is 2t.

The phase difference due to path length is (2π / λ) * 2t = 4πt / λ

Total phase difference = π (from reflection) + 4πt / λ

For constructive interference, total phase difference = 2πk

So:

π + 4πt / λ = 2πk

Divide both sides by π:

1 + 4t / λ = 2k

→ 4t / λ = 2k - 1

→ t = (2k - 1)λ / 4

For minimum thickness, k=1:

t = (2*1 - 1) * λ / 4 = λ / 4

So t = 600 nm / 4 = 150 nm

But wait — is the wavelength λ = 600 nm in air? Yes, as given.

And since the phase shift is accounted for, this is correct.

However, note that in some contexts, if the film is surrounded by media with different indices, you might need to consider the wavelength inside the film. But here, since we’re using the wavelength in air for the calculation, and the phase shifts are correctly accounted for, this should be fine.

Alternatively, we can think of it as:

The condition for constructive interference (bright reflection) when there is one phase reversal is:

> 2t = (m + ½)λ

This is standard.

Yes, so t = (m + ½)λ / 2

For m=0: t = λ/4 = 150 nm

---

### Step 4: Check if this makes sense

Is there any reason to use wavelength in glass? For example, if we were calculating interference in the film, we might use λ_glass = λ_air / n_glass.

But in this case, since we are asked for the thickness that causes reflection to be bright (orange), and we are told the wavelength in air, and the phase shifts are correctly handled, we don’t need to adjust for wavelength in glass.

Moreover, the standard formula uses λ_air for the condition, even though the actual wavelength inside the film is shorter.

So yes, t = λ/4 = 600 nm / 4 = 150 nm.

---

### Final Answer

Answer: C
```
