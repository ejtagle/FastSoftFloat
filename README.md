# SoftFloat library

This library is a soft float implementation optimized for speed for the ARM Cortex M3 based microcontrollers</br>
It is written in C++20, and is constexpr, so all calculations that can be done at compile time will be</br>
It also implements autodetection and optimization of fused multiplications and additions, to increase speed and accuracy</br>

All the library is in a header file called FusedSoftFloat.hh</br>

To use it, just include it, and, instead of using the 'float' type for variables, use the 'SoftFloat' type.</br>
It has equivalent resolution as of IEEE float type (29 bits mantissa, 7 bits exponent)</br>
No NaN, +/-Inf are handled and/or created</br>
</br>

# Expected approximate cycle counts

# SoftFloat Cycle Count Analysis (GD32F103 / Cortex‑M3)

## 1. SoftFloat Cycle Counts (min / typical / max)

| Function                           | Min | Typ | Max | Remarks                                                                 |
|------------------------------------|-----|-----|-----|-------------------------------------------------------------------------|
| Construction / assignment          |   5 |  10 |  30 | Normalisation dominates.                                                 |
| `operator+`, `operator-`           |   8 |  15 |  45 | Fast path: same sign addition; slow path: cancellation with CLZ.        |
| `operator*` (`mul_plain`)          |   6 |  12 |  20 | Single SMULL + shift + SSAT.                                             |
| `operator/`                        |  10 |  18 |  30 | Table‑based reciprocal + one multiply; no hardware division.             |
| `sqrt()`                           |  20 |  35 |  60 | Goldschmidt method: table lookup + 2‑3 multiplies + final multiply.     |
| `inv_sqrt()`                       |  15 |  25 |  40 | Table + one Newton iteration (2 multiplies).                             |
| `exp()`                            |  25 |  40 |  80 | SMULL for range reduction, table lookup, linear interpolation.           |
| `log2()`                           |  15 |  25 |  40 | Table lookup + interpolation; very fast.                                 |
| `log()` / `log10()`                |  20 |  30 |  50 | One extra multiply after `log2()`.                                       |
| `sincos()`                         |  30 |  60 | 150 | Table interpolation; range reduction may use UDIV or loop.               |
| `sin()` / `cos()`                  |  25 |  55 | 140 | Wrappers around `sincos()`.                                              |
| `tan()`                            |  40 |  80 | 200 | `sincos()` + reciprocal + multiply.                                      |
| `asin()` / `acos()`                |  50 | 100 | 200 | Uses `atan2()` + `sqrt()`; moderate cost.                                |
| `atan2()`                          |  40 |  90 | 180 | Table + series; UDIV used in some paths.                                 |
| `sinh()` / `cosh()` / `tanh()`     |  35 |  55 | 100 | Based on `exp()`.                                                        |
| `pow()` (integer exponent)         |   5 |  20 | 100 | Fast exponentiation by squaring; loop count up to 31.                    |
| `pow()` (general)                  |  80 | 150 | 400 | Calls `exp(y*log(x))`.                                                   |
| `trunc()` / `floor()` / `ceil()` / `round()` |   8 |  15 |  30 | Simple bit manipulations.                                                |
| `fmod()`                           |  15 |  40 | 200 | Iterative remainder with UDIV; loops proportional to exponent difference.|
| `fma()` / `fused_mul_add`          |  10 |  20 |  40 | 64‑bit product + addition.                                               |
| `hypot()`                          |  15 |  35 |  70 | One square root.                                                         |
| `lerp()`                           |  15 |  25 |  40 | One multiply + two adds.                                                 |

## 2. Comparison with Typical Soft‑Float Library and qfplib

| Function         | SoftFloat Typical | Soft‑float Lib Typical | qfplib Typical | Comments |
|------------------|-------------------|------------------------|----------------|----------|
| `fadd` / `fsub`  | 15                | 80–120                 | 35–50          | qfplib uses alignment shifts and careful rounding. |
| `fmul`           | 12                | 60–90                  | 20–30          | qfplib single SMULL + normalization. |
| `fdiv`           | 18                | 150–250                | 40–70          | qfplib uses UDIV for quotient estimate + correction. |
| `fsqrt`          | 35                | 300–500                | 50–80          | qfplib table + two Newton iterations with SDIV. |
| `fexp`           | 40                | 500–800                | 80–150         | qfplib uses table + power series. |
| `fln` / `flog`   | 30                | 400–700                | 60–100         | qfplib uses table + series. |
| `fsin` / `fcos`  | 55                | 400–700                | 100–200        | qfplib performs range reduction and table interpolation. |
| `ftan`           | 80                | 600–1000               | 150–250        | qfplib uses fraction decomposition. |
| `fatan2`         | 90                | 600–1000               | 100–200        | qfplib uses table + refinement. |

## Notes

- **Cortex‑M3 instruction timings** used for analysis:
  - Data processing: 1 cycle
  - Load/store: 2–3 cycles
  - Branches: 1–3 cycles (taken penalty)
  - Multiply (SMULL/UMULL): 1 cycle
  - Divide (UDIV/SDIV): 2–12 cycles (early‑out)
  - CLZ, SSAT/USAT: 1 cycle
- **SoftFloat** timings assume `__arm__` runtime path with branch hints.
- **qfplib** timings derived by static analysis of provided assembly.
- “Typical” values represent common‑case paths (normalized operands, moderate exponent differences).

### Performance
At this point, the library is nearly 8 times slower than hardware float support.
