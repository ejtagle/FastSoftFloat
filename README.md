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


### Results from running in a STM32F407 @ 168Mhz

##  SoftFloat Test Suite

=== FUNCTIONAL TESTS ===</br>
[PASS] Default Constructor</br>
[PASS] Integer Constructor</br>
[PASS] Float Constructor</br>
[PASS] Constants</br>
[PASS] Addition</br>
[PASS] Subtraction</br>
[PASS] Multiplication</br>
[PASS] Division</br>
[PASS] Negation</br>
[PASS] Absolute Value</br>
[PASS] Comparison</br>
[PASS] Shift Operators</br>
[PASS] Fused Multiply-Add</br>
[PASS] Fused Multiply-Sub</br>
[PASS] Fused Multiply-Multiply-Add</br>
[PASS] Square Root</br>
[PASS] Inverse Square Root</br>
[PASS] Sine</br>
[PASS] Cosine</br>
[PASS] Linear Interpolation</br>
[PASS] Min/Max</br>
[PASS] Clamp</br>
[PASS] Integer Conversion</br>
[PASS] Chained Operations</br>
[PASS] Overflow Handling</br>
[PASS] Underflow Handling</br>
[PASS] Edge Cases</br>
[PASS] Compound Assignment</br>
[PASS] Expression Templates</br>
[PASS] Fused Multiply-Sub (ext)</br>
[PASS] Fused Mul-Mul-Sub (ext)</br>
[PASS] Invariant Normalization</br>
[PASS] Zero Semantics</br>
[PASS] Div/Zero & Inf Inputs</br>
[PASS] Int Conv Edges</br>
[PASS] Exponent Saturation</br>
[PASS] Math Symmetries</br>
[PASS] Lerp Edges</br>
[PASS] Expr Negation</br>
[PASS] Constexpr Complex</br>
[PASS] Shift Edges</br>
[PASS] Comparison Logic</br>
[PASS] Corner Cases & Bit Patterns</br>
[PASS] atan2</br>
  atan2 worst case error: 0.00 degree</br>
[PASS] atan2 (v2)</br>
[PASS] exp</br>
[PASS] hypot</br>
[PASS] tan</br>
[PASS] asin/acos</br>
[PASS] log/log2/log10</br>
[PASS] pow</br>
[PASS] rounding</br>
[PASS] hyperbolic</br>
[PASS] copysign/fmod</br>
[PASS] Random Stress Test</br>

-----------------------------------------------------------
Results: 55/55 passed, 0 failed
-----------------------------------------------------------

=== BENCHMARKS ===
| Operation |             : |    timing     |
|-----------|:-------------:|--------------:|
| Multiply  | (100000 ops): |  184.58 ns/op |
| Add       | (100000 ops): |  250.75 ns/op |
| Subtract  | (100000 ops): |  375.10 ns/op |
| Divide    | (100000 ops): |  309.61 ns/op |
| FMA       | (100000 ops): |  809.71 ns/op |
| FMS       | (100000 ops): |  732.31 ns/op |
| FMMA      | (100000 ops): | 1000.22 ns/op |
| FMMS      | (100000 ops): | 1101.43 ns/op |
| sin       | (100000 ops): |  922.83 ns/op |
| cos       | (100000 ops): |  928.79 ns/op |
| tan       | (100000 ops): | 1274.19 ns/op |
| asin      | (100000 ops): | 2125.81 ns/op |
| acos      | (100000 ops): | 2063.13 ns/op |
| atan2     | (100000 ops): | 1274.08 ns/op |
| sinh      | (100000 ops): | 1452.69 ns/op |
| cosh      | (100000 ops): | 1452.69 ns/op |
| tanh      | (100000 ops): | 1732.51 ns/op |
| exp       | (100000 ops): |  726.36 ns/op |
| log       | (100000 ops): |  625.14 ns/op |
| log2      | (100000 ops): |  547.75 ns/op |
| log10     | (100000 ops): |  631.10 ns/op |
| pow       | (100000 ops): |  708.49 ns/op |
| sqrt      | (100000 ops): |  184.59 ns/op |
| inv_sqrt  | (100000 ops): |  214.35 ns/op |
| floor     | (100000 ops): |  303.66 ns/op |
| ceil      | (100000 ops): |  565.61 ns/op |
| trunc     | (100000 ops): |  321.52 ns/op |
| round     | (100000 ops): |  333.42 ns/op |
| fract     | (100000 ops): |  672.77 ns/op |
| modf      | (100000 ops): |  720.40 ns/op |
| abs       | (100000 ops): |   59.56 ns/op |
| copysign  | (100000 ops): |   59.56 ns/op |
| fmod      | (100000 ops): |  672.77 ns/op |
| hypot     | (100000 ops): |  476.75 ns/op |
| lerp      | (100000 ops): |  970.45 ns/op |
| compare < | (100000 ops): |  107.19 ns/op |
| compare = | (100000 ops): |   23.84 ns/op |
| shift <<  | (100000 ops): |   77.42 ns/op |
| shift >>  | (100000 ops): |   71.47 ns/op |
| to_float  | (100000 ops): |   59.56 ns/op |
| to_int32  | (100000 ops): |  101.23 ns/op |
| from_floa | (100000 ops): |   23.84 ns/op |
| from_int  | (100000 ops): |  196.49 ns/op |
| Dot4      | (100000 ops): | 2643.95 ns/op |

ALL TESTS PASSED


### Results for our SoftFloat implementation

LINPACK benchmark, SoftFloat precision.
Machine precision:  6 digits.
Array size 100 X 100.
Average rolled and unrolled performance:

| Reps | Time(s) | DGEFA  | DGESL | OVERHEAD | KFLOPS   |
|------|---------|--------|-------|----------|----------|
| 16   | 0.88    | 85.08% | 5.18% | 9.74%    | 3561.242 |
| 32   | 1.76    | 85.08% | 5.18% | 9.74%    | 3561.244 |
| 64   | 3.52    | 85.08% | 5.18% | 9.74%    | 3561.239 |
| 128  | 7.04    | 85.08% | 5.18% | 9.74%    | 3561.228 |

Calibrate
| 0.28 Seconds |  1 | Passes (x 100) |
| 1.38 Seconds |  5 | Passes (x 100) |
| 6.92 Seconds | 25 | Passes (x 100) |

Use 361  passes (x 100)

          SoftFloat Precision C/C++ Whetstone Benchmark

| Loop content      |         Result       | MFLOPS |  MOPS  | Seconds|
|-------------------|----------------------|--------|--------|--------|
| N1 floating point | -1.12444269657135010 | 39.048 |        |  0.178 |
| N2 floating point | -1.12243998050689697 |  2.567 |        | 18.898 |
| N3 if then else   |  1.00000000000000000 |        | 83.981 |  0.445 |
| N4 fixed point    | 12.00000000000000000 |        | 83.975 |  1.354 |
| N5 sin,cos etc.   |  0.49188536405563354 |        |  1.800 | 16.687 |
| N6 floating point |  0.99991601705551147 | 75.377 |        |  2.583 |
| N7 assignments    |  3.00000000000000000 |        | 71.985 |  0.927 |
| N8 exp,sqrt etc.  |  0.75000041723251343 |        |  2.624 |  5.117 |
|-------------------|----------------------|--------|--------|--------|
| MWIPS             |                      | 78.159 |        | 46.188 |

### Results for hardware float implementation

LINPACK benchmark, Single precision.
Machine precision:  6 digits.
Array size 100 X 100.
Average rolled and unrolled performance:

| Reps | Time(s) | DGEFA  | DGESL | OVERHEAD | KFLOPS    |
|------|---------|--------|-------|----------|-----------|
| 128  | 0.93    | 79.53% | 4.89% | 15.58%   | 28912.676 |
| 256  | 1.85    | 79.53% | 4.89% | 15.58%   | 28912.655 |
| 512  | 3.71    | 79.53% | 4.89% | 15.58%   | 28912.678 |
| 1024 | 7.41    | 79.53% | 4.89% | 15.58%   | 28912.311 |

Calibrate
| 0.05 Seconds |   1 | Passes (x 100) |
| 0.26 Seconds |   5 | Passes (x 100) |
| 1.32 Seconds |  25 | Passes (x 100) |
| 6.62 Seconds | 125 | Passes (x 100) |

Use 1889  passes (x 100)

          Single Precision C/C++ Whetstone Benchmark

| Loop content      |         Result       | MFLOPS |  MOPS  | Seconds|
|-------------------|----------------------|--------|--------|--------|
| N1 floating point | -1.12391686439514160 | 116.845|        |   0.310|
| N2 floating point | -1.12191414833068848 |  82.272|        |   3.086|
| N3 if then else   |  1.00000000000000000 |        | 100.779|   1.940|
| N4 fixed point    | 12.00000000000000000 |        |8670.000|   0.000|
| N5 sin,cos etc.   |  0.49909299612045288 |        |   9.914|  15.853|
| N6 floating point |  0.99999982118606567 |  62.988|        |  16.177|
| N7 assignments    |  3.00000000000000000 |        |  71.985|   4.849|
| N8 exp,sqrt etc.  |  0.75110864639282227 |        |  10.566|   6.651|
|-------------------|----------------------|--------|--------|--------|
| MWIPS             |                      | 386.563|        |  48.866|

### Conclusion

Our software floating point is 386.563 / 78.159 = 4.95 times slower than hardware floating point on the Whetstone benchmark and 28912.311 / 3561.228 = 8.11 times slower in the Linpack benchmark.

