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

| Function / Operator                     | Min | Typ | Max | Remarks |
|-----------------------------------------|-----|-----|-----|---------|
| **Construction / Assignment**           |     |     |     |         |
| `SoftFloat()` (default)                 | 1   | 1   | 1   | Trivial inline |
| `SoftFloat(int32_t)`                    | 8   | 10  | 25  | Normalisation: CLZ (1) + shift + SSAT (1). Worst‑case many left shifts. |
| `SoftFloat(float)`                      | 12  | 15  | 20  | Bit‑cast + IEEE unpack + normalisation. |
| `operator=(int32_t / float)`            | 8   | 10  | 25  | Same as constructor from same type. |
| **Conversions**                         |     |     |     |         |
| `to_float()`                            | 5   | 6   | 8   | Few integer ops + bit‑cast. No normalisation. |
| `to_int32()`                            | 5   | 7   | 12  | Shift + sign handling. |
| **Unary**                               |     |     |     |         |
| `operator-()`                           | 1   | 1   | 1   | Flip sign bit. |
| `abs()`                                 | 2   | 2   | 2   | Branchless absolute value. |
| **Addition / Subtraction**              |     |     |     |         |
| `operator+` (same exponent)             | 8   | 10  | 15  | Add + overflow check (EOR/TST/ASR). |
| `operator+` (different exponent)        | 10  | 12  | 20  | Alignment shift + add + normalisation. |
| `operator-` (same exponent)             | 8   | 12  | 20  | Subtract + cancellation normalisation (CLZ). |
| `operator-` (different exponent)        | 10  | 14  | 22  | Alignment + subtract + normalisation. |
| **Multiplication**                      |     |     |     |         |
| `operator*` (deferred via `sf_mul_expr`)| 0   | 0   | 0   | No evaluation; only creates proxy. |
| `SoftFloat::mul_plain` (materialised)   | 9   | 11  | 14  | SMULL (3‑5) + ORR/MOV + overflow fix‑up. |
| **Division**                            |     |     |     |         |
| `operator/` (runtime, Knuth D)          | 20  | 28  | 45  | Two UDIV (2‑12 each) + corrections. |
| `operator/` (fallback 64‑bit)           | 60  | 70  | 90  | `__aeabi_uldivmod` call. |
| **Shifts**                              |     |     |     |         |
| `operator<<` / `operator>>`             | 2   | 3   | 5   | Exponent adjust + saturation check. |
| **Comparisons**                         |     |     |     |         |
| `operator==` / `!=`                     | 2   | 2   | 2   | Compare mantissa & exponent. |
| `operator<` / `>` / `<=` / `>=`         | 6   | 8   | 10  | Sign + exponent + mantissa branching. |
| **Math Functions**                      |     |     |     |         |
| `sin()` / `cos()`                       | 45  | 55  | 80  | Range reduction + table lookup (256 entries) + interpolation. |
| `tan()`                                 | 50  | 60  | 90  | `sincos` + division. |
| `asin()` / `acos()`                     | 55  | 70  | 100 | `atan2` + `sqrt` (one extra division). |
| `atan2(y, x)`                           | 55  | 75  | 120 | Table + interpolation + octant logic; UDIV in range reduction. |
| `exp()`                                 | 35  | 45  | 65  | Range reduction (multiply + to_int32) + table interpolation. |
| `log()`                                 | 40  | 50  | 70  | `log2` + one multiply. |
| `log2()`                                | 35  | 45  | 60  | Table lookup (257 entries) + interpolation. |
| `log10()`                               | 40  | 50  | 70  | `log2` + multiply. |
| `pow(y)`                                | 70  | 100 | 150 | `exp(y*log(x))`; may use integer exponent fast path. |
| `sqrt()`                                | 20  | 28  | 40  | `inv_sqrt` + multiply. |
| `inv_sqrt()`                            | 18  | 25  | 35  | Table seed + two Newton iterations (each ~9‑11 cycles for mul+sub). |
| `hypot(x, y)`                           | 25  | 35  | 50  | Scaling + multiply + add + `sqrt`. |
| `sinh()` / `cosh()` / `tanh()`          | 50  | 65  | 90  | `exp` + division/shift. |
| **Rounding / Truncation**               |     |     |     |         |
| `trunc()`                               | 8   | 10  | 15  | Convert to int32 + construct back. |
| `floor()` / `ceil()`                    | 10  | 14  | 20  | Integer shift + sign logic. |
| `round()`                               | 8   | 12  | 18  | Add bias + shift. |
| `fract()`                               | 12  | 16  | 22  | `trunc` + subtract. |
| `modf()`                                | 14  | 18  | 25  | `trunc` + subtract, returns pair. |
| **Fused Operations**                    |     |     |     |         |
| `fused_mul_add(a, b, c)`                | 15  | 20  | 30  | Multiply (SMULL) + align + add + normalise. |
| `fused_mul_sub(a, b, c)`                | 15  | 20  | 30  | Same as add, product negated. |
| `fused_mul_mul_add(a,b,c,d)`            | 20  | 28  | 40  | Two multiplies + align + add + normalise. |
| `fused_mul_mul_sub(a,b,c,d)`            | 20  | 28  | 40  | Same as add, second product negated. |
| **Utilities**                           |     |     |     |         |
| `clamp(lo, hi)`                         | 12  | 16  | 20  | Two comparisons. |
| `copysign(sign)`                        | 2   | 2   | 2   | Conditional sign flip. |
| `fmod(y)`                               | 30  | 40  | 60  | Division + trunc + multiply + subtract. |
| `fma(b, c)`                             | 15  | 20  | 30  | Same as `fused_mul_add`. |
| `lerp(a, b, t)`                         | 20  | 28  | 40  | Subtract + multiply + add. |

### Performance
At this point, the library is exactly 8 times slower than hardware float support.
