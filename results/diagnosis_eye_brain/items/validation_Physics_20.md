# validation_Physics_20

- 과목: Physics / 유형: multiple-choice
- **정답: A**
- 모델 최종 답 (seed 3407 / 1 / 2): B / B / B
- seed 3407 추출 줄: `B`

## 이미지

**image 1** (360×270 px)

![image 1](../images/validation_Physics_20_1.png)

## 문제

<image 1>A student performs the photoelectric effect experiment and obtains the data depicted in the accompanying graph of E_{km} (maximum kinetic energy) of photoelectrons v. the frequency of the photons. What is the approximate work function of this material?

- **A.** 1.5 eV
- **B.** 2.0 eV
- **C.** 2.7 eV
- **D.** 3.5 eV

## 모델 응답 (seed 3407 전문)

```text
We are given a graph of the maximum kinetic energy (E_km) of photoelectrons versus the frequency (f) of incident photons. This is a standard photoelectric effect graph.

According to Einstein’s photoelectric equation:

E_km = hν - φ

where:
- E_km is the maximum kinetic energy of the emitted electrons,
- hν is the photon energy (h is Planck’s constant, ν is frequency),
- φ is the work function of the material (the minimum energy needed to eject an electron).

In a plot of E_km vs. ν, this is a straight line with slope = h (Planck’s constant), and the y-intercept is -φ.

However, in this graph, we don’t have a vertical intercept — instead, we see that the line crosses the x-axis (where E_km = 0) at approximately ν = 4 × 10^14 Hz.

That means when the photon frequency is 4 × 10^14 Hz, the kinetic energy of the photoelectrons is zero. This is the **threshold frequency** (ν_0), below which no photoelectrons are emitted.

At threshold frequency, the photon energy exactly equals the work function:

φ = hν_0

So, we need to find the value of φ by calculating hν_0.

But we don’t have h explicitly, so we can use the fact that the slope of the line gives us h, and the intercept gives us -φ.

Alternatively, since we know the graph passes through the point where E_km = 0 at ν = 4 × 10^14 Hz, and we can pick another point on the line to compute the slope, then use the equation to find φ.

Let’s pick two points from the graph:

Point 1: At ν = 4 × 10^14 Hz, E_km = 0 eV → (4, 0)

Point 2: At ν = 8 × 10^14 Hz, E_km = 2.0 eV → (8, 2.0)

Wait — let’s check: at f = 8 × 10^14 Hz, the graph shows E_km = 2.0 eV? Let’s look at the grid.

The x-axis is labeled "f in 10^14", so 8 corresponds to 8 × 10^14 Hz.

Looking at the y-axis, at f = 8 × 10^14 Hz, the E_km is about 2.0 eV (since it goes up to 2.0 and the line passes through (8,2)).

So, we have two points:

(ν₁, E_km₁) = (4 × 10¹⁴, 0 eV)

(ν₂, E_km₂) = (8 × 10¹⁴, 2.0 eV)

Now, the slope of the line is:

slope = ΔE_km / Δν = (2.0 - 0) / (8 × 10¹⁴ - 4 × 10¹⁴) = 2.0 / (4 × 10¹⁴) = 0.5 × 10⁻¹⁴ eV/Hz

But we don’t need the exact value of h — we just need to find φ.

From the equation:

E_km = hν - φ

At threshold frequency (ν₀ = 4 × 10¹⁴ Hz), E_km = 0:

0 = h * (4 × 10¹⁴) - φ → φ = h * (4 × 10¹⁴)

We can also write:

φ = hν₀

But we can calculate φ directly using the graph.

Another approach: Use the equation with the two points.

We can write:

E_km = hν - φ

So for any point (ν, E_km):

E_km + φ = hν

Therefore, φ = hν - E_km

But again, without h, we can't do this unless we know h.

Wait — we can use the fact that the y-intercept is -φ.

The y-intercept is the value of E_km when ν = 0.

From the graph, if we extend the line backward to where ν = 0, what is E_km?

The line has a positive slope. It crosses the x-axis at ν = 4 × 10¹⁴ Hz, meaning when ν = 0, E_km must be negative.

Let’s find the y-intercept.

We can use the two points to find the equation of the line.

Points: (4, 0) and (8, 2)

Slope m = (2 - 0)/(8 - 4) = 2/4 = 0.5 eV per 10¹⁴ Hz

So, equation: E_km = 0.5*(ν - 4)  [because when ν=4, E_km=0]

Wait, let’s write it properly.

General form: E_km = m*(ν - ν₀)

Where ν₀ = 4 × 10¹⁴ Hz

So E_km = 0.5 * (ν - 4) × 10¹⁴ ? No — units.

Actually, since the x-axis is in 10¹⁴ Hz, let’s define ν in units of 10¹⁴ Hz.

Let’s redefine:

Let ν' = ν / 10¹⁴ → dimensionless frequency unit.

Then, from the graph:

At ν' = 4, E_km = 0

At ν' = 8, E_km = 2.0

Slope = (2.0 - 0)/(8 - 4) = 0.5 eV per unit ν'

So E_km = 0.5 * (ν' - 4)

Thus, when ν' = 0 (i.e., ν = 0), E_km = 0.5*(0 - 4) = -2.0 eV

So the y-intercept is -2.0 eV.

But from the equation E_km = hν - φ, the y-intercept (when ν=0) is -φ.

Therefore, -φ = -2.0 eV → φ = 2.0 eV

That’s straightforward.

So the work function is 2.0 eV.

Answer: B
```
