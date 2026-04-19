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

## Basic Arithmetic

### ADDITION / SUBTRACTION
| Operation | SF min | SF typ | SF max |
|-----------|--------|--------|--------|
| operator+ | 8      | 45     | 70     |
| operator- | 10     | 48     | 75     |

### MULTIPLICATION
| Operation                            | SF min | SF typ | SF max |
|--------------------------------------|--------|--------|--------|
| operator\*                           | 18     | 38     | 58     |
| fused_mul_add(a,b,c) — a+b·c         | 18     | 55     | 90     |
| fused_mul_sub(a,b,c) — a−b·c         | 18     | 58     | 95     |
| fused_mul_mul_add(a,b,c,d) — a·b+c·d | 20     | 72     | 115    |

### DIVISION
| Operation | SF min | SF typ | SF max |
|-----------|--------|--------|--------|
| operator/ | 38     | 75     | 120    |

### Conversions & Scalar Utilities
| Operation              | SF min | SF typ | SF max |
|------------------------|--------|--------|--------|
| operator- (negate)     | 14     | 15     | 16     |
| abs()                  | 16     | 18     | 20     |
| operator<< (×2ⁿ scale) | 8      | 15     | 24     |
| operator>> (÷2ⁿ scale) | 8      | 13     | 18     |
| operator==             | 12     | 22     | 35     |
| operator<              | 12     | 28     | 45     |
| SoftFloat(int32_t)     | 8      | 24     | 38     |
| SoftFloat(float)       | 8      | 18     | 28     |
| to_float()             | 8      | 16     | 24     |
| to_int32()             | 8      | 18     | 30     |

## Transcendental & Math Functions

### SQUARE ROOT / INVERSE SQUARE ROOT
| Operation                   | SF min | SF typ | SF max |
|-----------------------------|--------|--------|--------|
| inv_sqrt() — Q-rsqrt+2×NR   | 70     | 130    | 200    |
| sqrt() — \*this · inv_sqrt()| 100    | 180    | 280    |

### TRIGONOMETRY (sin/cos: 512-entry table + 1-point FMA interp)
| Operation                             | SF min | SF typ | SF max |
|---------------------------------------|--------|--------|--------|
| sin()                                 | 100    | 170    | 250    |
| cos()                                 | 100    | 170    | 250    |
| sincos() — both for cost of one       | 100    | 168    | 248    |
| tan() — sincos + div                  | 180    | 255    | 380    |
| asin() — FMS + inv_sqrt + atan2       | 250    | 375    | 535    |
| acos() — asin chain                   | 265    | 395    | 555    |
| atan2() — div + 256-entry table + FMA | 90     | 138    | 202    |

### EXPONENTIAL / LOGARITHM (256-entry table + linear interp)
| Operation                  | SF min | SF typ | SF max |
|----------------------------|--------|--------|--------|
| exp()                      | 85     | 128    | 180    |
| log() — log2 × ln2         | 70     | 108    | 155    |
| log2()                     | 60     | 95     | 142    |
| log10() — log2 × log10(2)  | 70     | 108    | 155    |
| pow(x,y) — log + mul + exp | 225    | 345    | 500    |

### HYPERBOLIC
| Operation                      | SF min | SF typ | SF max |
|--------------------------------|--------|--------|--------|
| sinh() — exp + div             | 200    | 310    | 430    |
| cosh() — exp + div             | 200    | 310    | 430    |
| tanh() — exp(2x) + (e−1)/(e+1) | 175    | 280    | 390    |

### GEOMETRY HELPERS
| Operation                            | SF min | SF typ | SF max |
|--------------------------------------|--------|--------|--------|
| hypot(x,y) — scale + fmma + inv_sqrt | 55     | 155    | 275    |
| lerp(a,b,t) — fused_mul_add path     | 12     | 55     | 95     |

### Rounding & Remainder
| Operation      | SF min | SF typ | SF max |
|----------------|--------|--------|--------|
| trunc()        | 14     | 28     | 48     |
| floor()        | 16     | 55     | 92     |
| ceil()         | 16     | 55     | 92     |
| round()        | 22     | 80     | 138    |
| fract()        | 18     | 55     | 100    |
| modf()         | 20     | 58     | 105    |
| fmod(x,y)      | 80     | 145    | 180    |
| clamp(v,lo,hi) | 8      | 28     | 50     |


### Performance
At this point, the library is exactly 8 times slower than hardware float support.
