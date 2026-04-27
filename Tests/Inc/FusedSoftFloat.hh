/*
 * FusedSoftFloat library/header
 * 
 * Copyright (c) 2026 Eduardo José Tagle (ejtagle@hotmail.com)
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/** @file FusedSoftFloat.hh
 *  @brief SoftFloat optimised for Cortex-M3, ARMv7-M
 *
 *  Representation:
 *    value = mantissa * 2^exponent
 *    mantissa == 0  =>  zero (exponent is always 0 when mantissa == 0)
 *    mantissa != 0  =>  abs(mantissa) in [2^29, 2^30)
 *                      bit 29 set, bits 31:30 clear in abs(mantissa)
 */
#pragma once
#include <cstdint>
#include <cstring>
#include <climits>

#if __cplusplus >= 202002L
#   include <bit>   // std::bit_cast, std::countl_zero — both constexpr
#endif

#ifndef __arm__
#define SF_INT_EQUALS_INT32
#endif

// =========================================================================
// Platform detection
// =========================================================================
#if defined(__GNUC__) || defined(__clang__)
#   define SF_INLINE    __attribute__((always_inline)) inline
#   define SF_NOINLINE  __attribute__((noinline))
#   define SF_HOT       __attribute__((hot))
#   define SF_FLATTEN   __attribute__((flatten))
#   define SF_PURE      __attribute__((pure))
#   define SF_CONST     __attribute__((const))
#   define LIKELY(x)    __builtin_expect(!!(x), 1)
#   define UNLIKELY(x)  __builtin_expect(!!(x), 0)
#else
#   define SF_INLINE    inline
#   define SF_NOINLINE
#   define SF_HOT
#   define SF_FLATTEN
#   define SF_PURE
#   define SF_CONST
#   define LIKELY(x)    (x)
#   define UNLIKELY(x)  (x)
#endif

// =========================================================================
// Consteval detection (no extra #include needed — GCC/Clang built-in)
// =========================================================================
#if defined(__GNUC__) || defined(__clang__)
#   define SF_IS_CONSTEVAL() __builtin_is_constant_evaluated()
#else
#   define SF_IS_CONSTEVAL() false
#endif

class SoftFloat;
struct IntFractPair;
struct SinCosPair;

// =========================================================================
// SoftFloat class
// =========================================================================
class SoftFloat {
public:
	// ------------------------------------------------------------------
	// Nested types
	// ------------------------------------------------------------------
	struct MulExpr;

	// ------------------------------------------------------------------
	// Normalization invariants
	// ------------------------------------------------------------------
	static constexpr uint32_t MANT_MIN      = 0x20000000u; // 2^29
	static constexpr uint32_t MANT_MAX      = 0x3FFFFFFFu; // 2^30 - 1
	static constexpr uint32_t MANT_OVERFLOW = 0x40000000u; // 2^30
	static constexpr uint32_t MANT_TOP_TWO  = 0x60000000u; // bits 30:29 mask
	static constexpr int32_t  MANT_BITS     = 29;
	static constexpr int32_t  EXP_MIN       = -128;
	static constexpr int32_t  EXP_MAX       = 127;
	static constexpr int32_t  EXP_BIAS      = 127 + MANT_BITS; // 156 for float

	// ------------------------------------------------------------------
	// Bit-cast helper (C++20 or fallback)
	// ------------------------------------------------------------------
	template<typename To, typename From>
	[[nodiscard]] static constexpr SF_INLINE To bitcast(From v) noexcept
	{
		static_assert(sizeof(To) == sizeof(From), "bitcast: size mismatch");
#if __cplusplus >= 202002L
		return std::bit_cast<To>(v);
#else
		To r; __builtin_memcpy(&r, &v, sizeof(To)); return r;
#endif
	}

	// ------------------------------------------------------------------
	// Cortex-M3 primitives
	// ------------------------------------------------------------------
	[[nodiscard]] static constexpr SF_CONST SF_INLINE int clz(uint32_t x) noexcept
	{
		return __builtin_clz(x);
	}

	[[nodiscard]] static constexpr SF_CONST SF_INLINE uint32_t abs32(int32_t m) noexcept
	{
		uint32_t mask = static_cast<uint32_t>(m >> 31);
		return (static_cast<uint32_t>(m) ^ mask) - mask;
	}

	[[nodiscard]] static constexpr SF_INLINE int32_t sat_exp_fast(int32_t e) noexcept
	{
		return e;
	}

	[[nodiscard]] static constexpr SF_CONST SF_INLINE int32_t sat_exp(int32_t e) noexcept
	{
#if defined(__arm__)
		if (!SF_IS_CONSTEVAL()) {
			int32_t r;
			__asm__(
				"ssat %0, #8, %1\n\t"
				: "=r"(r)
				: "r"(e));
			return r;
		}
#endif
		if (e > EXP_MAX) return EXP_MAX;
		if (e < EXP_MIN) return EXP_MIN;
		return e;
	}

	static constexpr SF_INLINE void normalise_fast(int32_t& m, int32_t& e) noexcept;
	[[nodiscard]] static constexpr SF_CONST SF_INLINE uint32_t recip32(uint32_t b) noexcept;

	// ------------------------------------------------------------------
	// Data
	// ------------------------------------------------------------------
	int32_t mantissa;
	int32_t exponent;

private:
	// Bypass normalisation (caller guarantees invariant)
	[[nodiscard]] static constexpr SoftFloat from_raw_unchecked(int32_t m, int32_t e) noexcept {
		SoftFloat r; r.mantissa = m; r.exponent = e; return r;
	}

	[[nodiscard]] static constexpr SF_INLINE SoftFloat finish_addsub(int32_t rm, int32_t re) noexcept {
		if (UNLIKELY(rm == 0)) return zero();

		uint32_t ab = abs32(rm);

		if (LIKELY((ab & MANT_TOP_TWO) == MANT_MIN)) {
			return from_raw_unchecked(rm, sat_exp_fast(re));
		}

		if (ab & MANT_OVERFLOW) {
			rm >>= 1;
			re += 1;
			return from_raw_unchecked(rm, sat_exp_fast(re));
		}

		normalise_fast(rm, re);
		return from_raw_unchecked(rm, re);
	}

	[[nodiscard]] static constexpr uint64_t isqrt64(uint64_t n) noexcept {
		if (n < 2) return n;
		uint64_t lo = 1, hi = n >> 1;
		if (hi > 0xFFFFFFFFULL) hi = 0xFFFFFFFFULL;
		while (lo <= hi) {
			uint64_t mid = lo + ((hi - lo) >> 1);
			uint64_t sq = mid * mid;
			if (sq == n) return mid;
			if (sq < n) lo = mid + 1;
			else hi = mid - 1;
		}
		return hi;
	}

	[[nodiscard]] static constexpr SF_INLINE SF_FLATTEN
	SoftFloat mul_plain(SoftFloat a, SoftFloat b) noexcept
	{
		if (UNLIKELY(!a.mantissa || !b.mantissa)) return zero();

		int32_t re = a.exponent + b.exponent + MANT_BITS;

#if defined(__arm__)
		if (!SF_IS_CONSTEVAL()) {
			int32_t rm, lo_r, hi_r, tmp;

			__asm__(
				"smull  %[lo], %[hi], %[am], %[bm]\n\t"
				"mov    %[rm], %[hi], lsl #3\n\t"
				"orr    %[rm], %[rm], %[lo], lsr #29\n\t"
				"asrs   %[tp], %[rm], #31\n\t"
				"eors   %[tp], %[rm], %[tp]\n\t"
				"subs   %[tp], %[tp], %[rm], asr #31\n\t"
				"cmp    %[tp], #0x40000000\n\t"
				"itt    cs\n\t"
				"asrcs  %[rm], %[rm], #1\n\t"
				"addcs  %[re], %[re], #1\n\t"
				: [rm] "=&r" (rm),
				  [lo] "=&r" (lo_r),
				  [hi] "=&r" (hi_r),
				  [tp] "=&r" (tmp),
				  [re] "+r"  (re)
				: [am] "r"   (a.mantissa),
				  [bm] "r"   (b.mantissa)
				: "cc");

			return from_raw_unchecked(rm, re);
		}
#endif
		{
			int64_t  prod  = static_cast<int64_t>(a.mantissa)
			               * static_cast<int64_t>(b.mantissa);
			int32_t  rm    = static_cast<int32_t>(prod >> MANT_BITS);
			uint32_t abs_m = abs32(rm);

			if (UNLIKELY(abs_m >= MANT_OVERFLOW)) {
				rm >>= 1;
				re  += 1;
			}
			return from_raw_unchecked(rm, re);
		}
	}

	constexpr SF_HOT void from_float(float f) noexcept {
		uint32_t bits = bitcast<uint32_t>(f);
		if ((bits & 0x7FFFFFFFu) == 0) { mantissa = 0; exponent = 0; return; }
		bool     neg = (bits >> 31) != 0;
		uint32_t expf = (bits >> 23) & 0xFFu;
		uint32_t frac = bits & 0x7FFFFFu;
		if (expf == 0xFFu) {
			mantissa = neg ? -static_cast<int32_t>(MANT_MIN) : static_cast<int32_t>(MANT_MIN);
			exponent = 98;
			return;
		}
		if (expf == 0) { mantissa = 0; exponent = 0; return; }
		uint32_t m = (1u << MANT_BITS) | (frac << 6);
		mantissa = neg ? -static_cast<int32_t>(m) : static_cast<int32_t>(m);
		exponent = static_cast<int32_t>(expf) - EXP_BIAS;
	}

	// =========================================================================
	// INV_SQRT_Q29 table definition
	// =========================================================================
	static constexpr int32_t INV_SQRT_Q29[257] = {
		0x20000000,	0x1FF00BF6,	0x1FE02FB0,	0x1FD06AF4,
		0x1FC0BD88,	0x1FB12733,	0x1FA1A7BB,	0x1F923EEA,
		0x1F82EC88,	0x1F73B05F,	0x1F648A3A,	0x1F5579E4,
		0x1F467F28,	0x1F3799D3,	0x1F28C9B3,	0x1F1A0E95,
		0x1F0B6849,	0x1EFCD69C,	0x1EEE595E,	0x1EDFF061,
		0x1ED19B76,	0x1EC35A6D,	0x1EB52D18,	0x1EA7134C,
		0x1E990CDB,	0x1E8B1998,	0x1E7D3959,	0x1E6F6BF2,
		0x1E61B139,	0x1E540903,	0x1E467328,	0x1E38EF7F,
		0x1E2B7DDE,	0x1E1E1E1E,	0x1E10D017,	0x1E0393A3,
		0x1DF6689B,	0x1DE94ED9,	0x1DDC4637,	0x1DCF4E8F,
		0x1DC267BE,	0x1DB5919F,	0x1DA8CC0E,	0x1D9C16E8,
		0x1D8F7209,	0x1D82DD4F,	0x1D765897,	0x1D69E3C0,
		0x1D5D7EA9,	0x1D512930,	0x1D44E334,	0x1D38AC96,
		0x1D2C8535,	0x1D206CF1,	0x1D1463AC,	0x1D086947,
		0x1CFC7DA3,	0x1CF0A0A2,	0x1CE4D225,	0x1CD91210,
		0x1CCD6046,	0x1CC1BCA9,	0x1CB6271C,	0x1CAA9F85,
		0x1C9F25C5,	0x1C93B9C4,	0x1C885B63,	0x1C7D0A8A,
		0x1C71C71C,	0x1C669100,	0x1C5B681B,	0x1C504C54,
		0x1C453D91,	0x1C3A3BB8,	0x1C2F46B0,	0x1C245E61,
		0x1C198AB3,	0x1C0EB38C,	0x1C03F0D5,	0x1BF93A75,
		0x1BEE9057,	0x1BE3F261,	0x1BD9607E,	0x1BCEDA96,
		0x1BC46093,	0x1BB9F25E,	0x1BAF8FE1,	0x1BA53907,
		0x1B9AEDBA,	0x1B90ADE4,	0x1B867970,	0x1B7C504A,
		0x1B72325B,	0x1B681F91,	0x1B5E17D5,	0x1B541B15,
		0x1B4A293C,	0x1B404236,	0x1B3665F0,	0x1B2C9457,
		0x1B22CD57,	0x1B1910DD,	0x1B0F5ED6,	0x1B05B730,
		0x1AFC19D8,	0x1AF286BC,	0x1AE8FDCB,	0x1ADF7EF1,
		0x1AD60A1D,	0x1ACC9F3E,	0x1AC33E42,	0x1AB9E718,
		0x1AB099AE,	0x1AA755F5,	0x1A9E1BDB,	0x1A94EB4F,
		0x1A8BC441,	0x1A82A6A2,	0x1A79925F,	0x1A70876B,
		0x1A6785B4,	0x1A5E8D2B,	0x1A559DC1,	0x1A4CB766,
		0x1A43DA0B,	0x1A3B05A0,	0x1A323A17,	0x1A297761,
		0x1A20BD70,	0x1A180C34,	0x1A0F639F,	0x1A06C3A3,
		0x19FE2C31,	0x19F59D3C,	0x19ED16B6,	0x19E49890,
		0x19DC22BE,	0x19D3B531,	0x19CB4FDD,	0x19C2F2B3,
		0x19BA9DA7,	0x19B250AB,	0x19AA0BB3,	0x19A1CEB1,
		0x19999999,	0x19916D5F,	0x198946F5,	0x19812950,
		0x19791363,	0x19710521,	0x1968FE80,	0x1960FF72,
		0x195907EB,	0x195117E1,	0x19492F47,	0x19414E12,
		0x19397436,	0x1931A1A8,	0x1929D65D,	0x19221249,
		0x191A5561,	0x19129F9B,	0x190AF0EA,	0x19034946,
		0x18FBA8A1,	0x18F40EF4,	0x18EC7C31,	0x18E4F050,
		0x18DD6B45,	0x18D5ED07,	0x18CE758B,	0x18C704C6,
		0x18BF9AB0,	0x18B8373E,	0x18B0DA66,	0x18A9841E,
		0x18A2345D,	0x189AEB18,	0x1893A847,	0x188C6BE0,
		0x188535D9,	0x187E0629,	0x1876DCCF,	0x186FB9AA,
		0x18689CC8,	0x18618618,	0x185A7592,	0x18536B2D,
		0x184C66DF,	0x184568A0,	0x183E7067,	0x18377E2C,
		0x183091E6,	0x1829AB8D,	0x1822CB18,	0x181BF07E,
		0x18151BB8,	0x180E4CBD,	0x18078386,	0x1800C009,
		0x17FA023F,	0x17F34A20,	0x17EC97A4,	0x17E5EAC3,
		0x17DF4375,	0x17D8A1B3,	0x17D20575,	0x17CB6EB3,
		0x17C4DD66,	0x17BE5186,	0x17B7CB0C,	0x17B149F0,
		0x17AACE2B,	0x17A457B5,	0x179DE689,	0x17977A9D,
		0x179113EB,	0x178AB26D,	0x1784561B,	0x177EFEED,
		0x177AACDE,	0x17715FE6,	0x176B17FF,	0x1764D521,
		0x175E9746,	0x17585E68,	0x17522A7F,	0x174BFB85,
		0x1745D174,	0x173FAC45,	0x17398BF2,	0x17337073,
		0x172D59C4,	0x172747DD,	0x172139B9,	0x171B3251,
		0x17152E9F,	0x170F2F9D,	0x17093544,	0x17033F90,
		0x16FD4E79,	0x16F761FA,	0x16F17A0D,	0x16EB96AC,
		0x16E5B7D1,	0x16DFDD77,	0x16DA0797,	0x16D4362D,
		0x16CE6932,	0x16C8A0A0,	0x16C2DC73,	0x16BD1CA4,
		0x16B7612F,	0x16B1AA0D,	0x16ABF739,	0x16A648AE,
		0x16A09E66,
	};

public:
	// ------------------------------------------------------------------
	// Default constructor — zero
	// ------------------------------------------------------------------
	constexpr SoftFloat() noexcept : mantissa{ 0 }, exponent{ 0 } {}

	// ------------------------------------------------------------------
	// Normalising constructors
	// ------------------------------------------------------------------
	constexpr SF_HOT SoftFloat(int32_t m, int32_t e) noexcept
		: mantissa{ m }, exponent{ e }
	{
		normalise();
	}

#ifndef SF_INT_EQUALS_INT32
	constexpr SF_HOT explicit SoftFloat(int v) noexcept
	: mantissa{ v }, exponent{ 0 }
	{
		normalise();
	}
#endif

	constexpr SF_HOT explicit SoftFloat(int32_t v) noexcept
		: mantissa{ v }, exponent{ 0 }
	{
		normalise();
	}

	constexpr SF_HOT explicit SoftFloat(int16_t v) noexcept
		: mantissa{ static_cast<int32_t>(v) }, exponent{ 0 }
	{
		normalise();
	}

	constexpr SF_HOT explicit SoftFloat(float f) noexcept
		: mantissa{ 0 }, exponent{ 0 }
	{
		from_float(f);
	}

#ifndef SF_INT_EQUALS_INT32
	constexpr SF_HOT SoftFloat& operator=(int v) noexcept {
		mantissa = v;
		exponent = 0;
		normalise();
		return *this;
	}
#endif

	constexpr SF_HOT SoftFloat& operator=(int32_t v) noexcept {
		mantissa = v;
		exponent = 0;
		normalise();
		return *this;
	}

	constexpr SF_HOT SoftFloat& operator=(int16_t v) noexcept {
		mantissa = static_cast<int32_t>(v);
		exponent = 0;
		normalise();
		return *this;
	}

	constexpr SF_HOT SoftFloat& operator=(float f) noexcept {
		mantissa = 0;
		exponent = 0;
		from_float(f);
		return *this;
	}

	// Proxy constructor (defined after MulExpr)
	constexpr SF_HOT SoftFloat(const MulExpr& m) noexcept;

	// ------------------------------------------------------------------
	// Manual re-normalise
	//
	// Invariant: mantissa == 0  =>  zero
	//            mantissa != 0  =>  abs(mantissa) in [2^29, 2^30)
	// ------------------------------------------------------------------
	constexpr SF_HOT SF_INLINE void normalise() noexcept
	{
		int32_t m = mantissa, e = exponent;
		if (m == 0) { exponent = 0; return; }

		if (SF_IS_CONSTEVAL()) {
			uint32_t a = abs32(m);
			int lz    = clz(a);
			int shift = lz - 2;
			if (shift > 0) {
				int ne = e - shift;
				if (ne < -250) { mantissa = 0; exponent = 0; return; }
				a <<= shift; e = ne;
			}
			else if (shift < 0) {
				int rs = -shift; a >>= rs; e += rs;
			}
			exponent = sat_exp(e);
			mantissa = (m < 0) ? -static_cast<int32_t>(a) : static_cast<int32_t>(a);
			return;
		}

#if defined(__arm__)
		{
			uint32_t a;
			uint32_t lz;
			__asm__(
				"eor  %[a],  %[m], %[m], asr #31\n\t"
				"sub  %[a],  %[a], %[m], asr #31\n\t"
				"clz  %[lz], %[a]               \n\t"
				: [a]  "=&r" (a),
				  [lz] "=&r" (lz)
				: [m]  "r"   (m)
				: "cc");
			int32_t shift = static_cast<int32_t>(lz) - 2;
			if (shift > 0) {
				int ne = e - shift;
				if (UNLIKELY(ne < -250)) { mantissa = 0; exponent = 0; return; }
				a <<= shift; e = ne;
			}
			else if (shift < 0) {
				int rs = -shift; a >>= rs; e += rs;
			}
			__asm__(
				"ssat %0, #8, %1"
				: "=r"(e)
				: "r"(e));
			exponent = e;
			mantissa = (m < 0) ? -static_cast<int32_t>(a) : static_cast<int32_t>(a);
		}
#else
		{
			uint32_t a = abs32(m);
			int lz    = clz(a);
			int shift = lz - 2;
			if (shift > 0) {
				int ne = e - shift;
				if (ne < -250) { mantissa = 0; exponent = 0; return; }
				a <<= shift; e = ne;
			}
			else if (shift < 0) {
				int rs = -shift; a >>= rs; e += rs;
			}
			exponent = sat_exp(e);
			mantissa = (m < 0) ? -static_cast<int32_t>(a) : static_cast<int32_t>(a);
		}
#endif
	}

	// ------------------------------------------------------------------
	// to_float — constexpr in C++20 via std::bit_cast
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SF_HOT float to_float() const noexcept {
		if (!mantissa) return 0.f;
		uint32_t a = abs32(mantissa);
		int      iexp = exponent + EXP_BIAS;
		if (iexp >= 255) return mantissa > 0 ? 3.4028235e38f : -3.4028235e38f;
		if (iexp <= 0) return 0.f;
		uint32_t bits = (mantissa < 0 ? 0x80000000u : 0u)
			| (static_cast<uint32_t>(iexp) << 23)
			| ((a >> 6) & 0x007FFFFFu);
		return bitcast<float>(bits);
	}
	[[nodiscard]] constexpr explicit operator float()   const noexcept { return to_float(); }

	// ------------------------------------------------------------------
	// to_int32 — truncate toward zero, constexpr
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SF_HOT int32_t to_int32() const noexcept {
		if (!mantissa) return 0;
		if (exponent >= 2) return mantissa > 0 ? INT32_MAX : INT32_MIN;

		uint32_t a = abs32(mantissa);

		if (exponent >= 0) {
			a <<= exponent;
		}
		else {
			int rs = -exponent;
			if (rs >= 31) return 0;
			a >>= rs;
		}

		return mantissa < 0 ? -static_cast<int32_t>(a) : static_cast<int32_t>(a);
	}
	[[nodiscard]] constexpr explicit operator int32_t() const noexcept { return to_int32(); }

	// ------------------------------------------------------------------
	// Unary
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SoftFloat operator-() const noexcept { return from_raw_unchecked(-mantissa, exponent); }
	[[nodiscard]] constexpr SoftFloat operator+() const noexcept { return *this; }

	// ------------------------------------------------------------------
	// Binary operator declarations (defined after MulExpr)
	// ------------------------------------------------------------------
	friend constexpr SF_HOT SF_INLINE MulExpr operator*(SoftFloat a, SoftFloat b) noexcept;
	friend constexpr SF_HOT SF_INLINE MulExpr operator*(SoftFloat a, float     b) noexcept;
#ifndef SF_INT_EQUALS_INT32
	friend constexpr SF_HOT SF_INLINE MulExpr operator*(SoftFloat a, int       b) noexcept;
#endif
	friend constexpr SF_HOT SF_INLINE MulExpr operator*(SoftFloat a, int32_t   b) noexcept;
	friend constexpr SF_HOT SF_INLINE MulExpr operator*(float     a, SoftFloat b) noexcept;
#ifndef SF_INT_EQUALS_INT32
	friend constexpr SF_HOT SF_INLINE MulExpr operator*(int       a, SoftFloat b) noexcept;
#endif
	friend constexpr SF_HOT SF_INLINE MulExpr operator*(int32_t   a, SoftFloat b) noexcept;

	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
	friend SoftFloat operator+(SoftFloat a, SoftFloat b) noexcept
	{
		if (UNLIKELY(!a.mantissa)) return b;
		if (UNLIKELY(!b.mantissa)) return a;

		int d = a.exponent - b.exponent;
		if (d >= 31) return a;
		if (d <= -31) return b;

		int32_t rm, re;

		if (LIKELY(d == 0)) {
			re = a.exponent;

			if ((a.mantissa ^ b.mantissa) >= 0) {
				rm = a.mantissa + b.mantissa;
				rm >>= 1;
				if (UNLIKELY(++re > EXP_MAX)) re = EXP_MAX;
				return from_raw_unchecked(rm, re);
			}

			rm = a.mantissa + b.mantissa;
			if (UNLIKELY(rm == 0)) return zero();
			normalise_fast(rm, re);
			return from_raw_unchecked(rm, re);
		}

		if (d > 0) {
			rm = a.mantissa + (b.mantissa >> d);
			re = a.exponent;
		}
		else {
			rm = (a.mantissa >> -d) + b.mantissa;
			re = b.exponent;
		}

		if (UNLIKELY(rm == 0)) return zero();

		uint32_t ab = abs32(rm);
		if (LIKELY((ab & MANT_TOP_TWO) == MANT_MIN)) {
			return from_raw_unchecked(rm, re);
		}
		if (ab & MANT_OVERFLOW) {
			rm >>= 1;
			if (UNLIKELY(++re > EXP_MAX)) re = EXP_MAX;
			return from_raw_unchecked(rm, re);
		}
		normalise_fast(rm, re);
		return from_raw_unchecked(rm, re);
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator+(SoftFloat a, float b) noexcept {
		return a + SoftFloat(b);
	}
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator+(SoftFloat a, int b) noexcept {
		return a + SoftFloat(b);
	}
#endif
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator+(SoftFloat a, int32_t b) noexcept {
		return a + SoftFloat(b);
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator+(float a, SoftFloat b) noexcept {
		return SoftFloat(a) + b;
	}
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator+(int a, SoftFloat b) noexcept {
		return SoftFloat(a) + b;
	}
#endif
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator+(int32_t a, SoftFloat b) noexcept {
		return SoftFloat(a) + b;
	}

	// ------------------------------------------------------------------
	// Sub — constexpr
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
	friend SoftFloat operator-(SoftFloat a, SoftFloat b) noexcept
	{
		if (UNLIKELY(!b.mantissa)) return a;
		if (UNLIKELY(!a.mantissa)) return -b;

		int d = a.exponent - b.exponent;
		if (d >= 31) return a;
		if (d <= -31) return -b;

		int32_t rm, re;

		if (LIKELY(d == 0)) {
			re = a.exponent;

			if ((a.mantissa ^ b.mantissa) < 0) {
				rm = a.mantissa - b.mantissa;
				rm >>= 1;
				if (UNLIKELY(++re > EXP_MAX)) re = EXP_MAX;
				return from_raw_unchecked(rm, re);
			}

			rm = a.mantissa - b.mantissa;
			if (UNLIKELY(rm == 0)) return zero();
			normalise_fast(rm, re);
			return from_raw_unchecked(rm, re);
		}

		if (d > 0) {
			rm = a.mantissa - (b.mantissa >> d);
			re = a.exponent;
		}
		else {
			rm = (a.mantissa >> -d) - b.mantissa;
			re = b.exponent;
		}

		if (UNLIKELY(rm == 0)) return zero();

		uint32_t ab = abs32(rm);
		if (LIKELY((ab & MANT_TOP_TWO) == MANT_MIN)) {
			return from_raw_unchecked(rm, re);
		}
		if (ab & MANT_OVERFLOW) {
			rm >>= 1;
			if (UNLIKELY(++re > EXP_MAX)) re = EXP_MAX;
			return from_raw_unchecked(rm, re);
		}
		normalise_fast(rm, re);
		return from_raw_unchecked(rm, re);
	}

	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator-(SoftFloat a, float b) noexcept {
		return a - SoftFloat(b);
	}
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator-(SoftFloat a, int b) noexcept {
		return a - SoftFloat(b);
	}
#endif
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator-(SoftFloat a, int32_t b) noexcept {
		return a - SoftFloat(b);
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator-(float a, SoftFloat b) noexcept {
		return SoftFloat(a) - b;
	}
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator-(int a, SoftFloat b) noexcept {
		return SoftFloat(a) - b;
	}
#endif
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator-(int32_t a, SoftFloat b) noexcept {
		return SoftFloat(a) - b;
	}

	// =========================================================================
	// operator/ — unified: recip32 + one multiply
	// =========================================================================
	[[nodiscard]] constexpr SF_HOT SoftFloat operator/(SoftFloat rhs) const noexcept
	{
		if (UNLIKELY(!rhs.mantissa))
			return from_raw_unchecked(mantissa >= 0 ? MANT_MIN : -static_cast<int32_t>(MANT_MIN), EXP_MAX);
		if (UNLIKELY(!mantissa)) return zero();

		bool     neg = (mantissa ^ rhs.mantissa) < 0;
		uint32_t ua  = abs32(mantissa);
		uint32_t ub  = abs32(rhs.mantissa);

		uint32_t Y = recip32(ub);

		uint32_t qm;
		int32_t  qe = exponent - rhs.exponent - (MANT_BITS + 1);

#if defined(__arm__)
		if (!SF_IS_CONSTEVAL()) {
			uint32_t lo, hi;
			__asm__(
				"umull %0, %1, %2, %3"
				: "=&r"(lo),
				  "=&r"(hi)
				: "r"(ua),
				  "r"(Y));
			qm = (hi << 2) | (lo >> 30);
		}
		else
#endif
		{
			uint64_t q64 = static_cast<uint64_t>(ua) * static_cast<uint64_t>(Y);
			qm = static_cast<uint32_t>(q64 >> 30);
		}

		uint32_t rshift = qm >> (MANT_BITS + 1);
		uint32_t lshift = (~qm >> MANT_BITS) & 1u & ~rshift;
		qm = (qm >> rshift) << lshift;
		qe += static_cast<int32_t>(rshift) - static_cast<int32_t>(lshift);

		qe = sat_exp(qe);
		int32_t qm_signed = neg ? -static_cast<int32_t>(qm) : static_cast<int32_t>(qm);
		return from_raw_unchecked(qm_signed, qe);
	}

	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		SoftFloat operator/(float rhs) const noexcept {
		return *this / SoftFloat(rhs);
	}
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		SoftFloat operator/(int rhs) const noexcept {
		return *this / SoftFloat(rhs);
	}
#endif
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		SoftFloat operator/(int32_t rhs) const noexcept {
		return *this / SoftFloat(rhs);
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator/(float lhs, SoftFloat rhs) noexcept {
		return SoftFloat(lhs) / rhs;
	}
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator/(int lhs, SoftFloat rhs) noexcept {
		return SoftFloat(lhs) / rhs;
	}
#endif
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator/(int32_t lhs, SoftFloat rhs) noexcept {
		return SoftFloat(lhs) / rhs;
	}

	// ------------------------------------------------------------------
	// reciprocal — 1/x via recip32, O(1)
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SF_HOT SoftFloat reciprocal() const noexcept
	{
		if (UNLIKELY(!mantissa))
			return from_raw_unchecked(mantissa >= 0 ? MANT_MIN : -static_cast<int32_t>(MANT_MIN), EXP_MAX);

		bool     neg = mantissa < 0;
		uint32_t ua  = abs32(mantissa);
		uint32_t Y   = recip32(ua);

		uint32_t qm = Y >> 1;
		int32_t  qe = -59 - exponent;

		if (UNLIKELY(qm & MANT_OVERFLOW)) {
			qm >>= 1;
			qe += 1;
		}
		qe = sat_exp(qe);
		return from_raw_unchecked(neg ? -static_cast<int32_t>(qm) : static_cast<int32_t>(qm), qe);
	}

	// ------------------------------------------------------------------
	// Power-of-2 scaling (exponent adjust only, O(1)) — constexpr
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		SoftFloat operator>>(int s) const noexcept {
		if (UNLIKELY(!mantissa)) return zero();
		int32_t ne = exponent - s;
		if (UNLIKELY(ne < -250)) return zero();
		return from_raw_unchecked(mantissa, ne);
	}
	constexpr SF_HOT SF_INLINE SF_FLATTEN
		SoftFloat& operator>>=(int s) noexcept {
		if (UNLIKELY(!mantissa)) return *this;
		int32_t ne = exponent - s;
		if (UNLIKELY(ne < -250)) {
			mantissa = 0;
			exponent = 0;
			return *this;
		}
		exponent = ne;
		return *this;
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		SoftFloat operator<<(int s) const noexcept {
		if (UNLIKELY(!mantissa)) return zero();
		int32_t ne = exponent + s;
		if (UNLIKELY(ne > EXP_MAX))
			return from_raw_unchecked(mantissa > 0 ? MANT_MIN : -static_cast<int32_t>(MANT_MIN), EXP_MAX);
		return from_raw_unchecked(mantissa, ne);
	}
	constexpr SF_HOT SF_INLINE SF_FLATTEN
		SoftFloat& operator<<=(int s) noexcept {
		if (UNLIKELY(!mantissa)) return *this;
		int32_t ne = exponent + s;
		if (UNLIKELY(ne > EXP_MAX)) {
			mantissa = mantissa > 0 ? MANT_MIN : -static_cast<int32_t>(MANT_MIN);
			exponent = EXP_MAX;
			return *this;
		}
		exponent = ne;
		return *this;
	}

	// ------------------------------------------------------------------
	// Comparison — constexpr
	// ------------------------------------------------------------------
	[[nodiscard]] friend constexpr bool operator==(SoftFloat a, SoftFloat b) noexcept {
		return a.mantissa == b.mantissa && a.exponent == b.exponent;
	}
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] friend constexpr SF_INLINE bool operator==(int av, SoftFloat b) noexcept { return SoftFloat(av) == b; }
#endif
	[[nodiscard]] friend constexpr SF_INLINE bool operator==(int32_t av, SoftFloat b) noexcept { return SoftFloat(av) == b; }
	[[nodiscard]] friend constexpr SF_INLINE bool operator==(float av, SoftFloat b) noexcept { return SoftFloat(av) == b; }
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] friend constexpr SF_INLINE bool operator==(SoftFloat a, int bv) noexcept { return a == SoftFloat(bv); }
#endif
	[[nodiscard]] friend constexpr SF_INLINE bool operator==(SoftFloat a, int32_t bv) noexcept { return a == SoftFloat(bv); }
	[[nodiscard]] friend constexpr SF_INLINE bool operator==(SoftFloat a, float bv) noexcept { return a == SoftFloat(bv); }

	[[nodiscard]] friend constexpr SF_INLINE bool operator!=(SoftFloat a, SoftFloat b) noexcept { return !(a == b); }
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] friend constexpr SF_INLINE bool operator!=(int a, SoftFloat b) noexcept { return !(a == b); }
#endif
	[[nodiscard]] friend constexpr SF_INLINE bool operator!=(int32_t a, SoftFloat b) noexcept { return !(a == b); }
	[[nodiscard]] friend constexpr SF_INLINE bool operator!=(float a, SoftFloat b) noexcept { return !(a == b); }
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] friend constexpr SF_INLINE bool operator!=(SoftFloat a, int b) noexcept { return !(a == b); }
#endif
	[[nodiscard]] friend constexpr SF_INLINE bool operator!=(SoftFloat a, int32_t b) noexcept { return !(a == b); }
	[[nodiscard]] friend constexpr SF_INLINE bool operator!=(SoftFloat a, float b) noexcept { return !(a == b); }

	[[nodiscard]] friend constexpr bool operator< (SoftFloat a, SoftFloat b) noexcept {
		if (!a.mantissa) return b.mantissa > 0;
		if (!b.mantissa) return a.mantissa < 0;
		bool an = a.mantissa < 0, bn = b.mantissa < 0;
		if (an != bn) return an;
		if (a.exponent != b.exponent)
			return an ? a.exponent > b.exponent : a.exponent < b.exponent;
		return an ? a.mantissa > b.mantissa : a.mantissa < b.mantissa;
	}
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] friend constexpr SF_INLINE bool operator< (int av, SoftFloat b) noexcept { return SoftFloat(av) < b; }
#endif
	[[nodiscard]] friend constexpr SF_INLINE bool operator< (int32_t av, SoftFloat b) noexcept { return SoftFloat(av) < b; }
	[[nodiscard]] friend constexpr SF_INLINE bool operator< (float av, SoftFloat b) noexcept { return SoftFloat(av) < b; }
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] friend constexpr SF_INLINE bool operator< (SoftFloat a, int bv)     noexcept { return a < SoftFloat(bv); }
#endif
	[[nodiscard]] friend constexpr SF_INLINE bool operator< (SoftFloat a, int32_t bv)     noexcept { return a < SoftFloat(bv); }
	[[nodiscard]] friend constexpr SF_INLINE bool operator< (SoftFloat a, float bv)   noexcept { return a < SoftFloat(bv); }
	[[nodiscard]] friend constexpr SF_INLINE bool operator> (SoftFloat a, SoftFloat b) noexcept { return b < a; }
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] friend constexpr SF_INLINE bool operator> (int a, SoftFloat b) noexcept { return b < a; }
#endif
	[[nodiscard]] friend constexpr SF_INLINE bool operator> (int32_t a, SoftFloat b) noexcept { return b < a; }
	[[nodiscard]] friend constexpr SF_INLINE bool operator> (float a, SoftFloat b) noexcept { return b < a; }
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] friend constexpr SF_INLINE bool operator> (SoftFloat a, int b) noexcept { return b < a; }
#endif
	[[nodiscard]] friend constexpr SF_INLINE bool operator> (SoftFloat a, int32_t b) noexcept { return b < a; }
	[[nodiscard]] friend constexpr SF_INLINE bool operator> (SoftFloat a, float b) noexcept { return b < a; }

	[[nodiscard]] friend constexpr SF_INLINE bool operator<=(SoftFloat a, SoftFloat b) noexcept { return !(a > b); }
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] friend constexpr SF_INLINE bool operator<=(int a, SoftFloat b) noexcept { return !(a > b); }
#endif
	[[nodiscard]] friend constexpr SF_INLINE bool operator<=(int32_t a, SoftFloat b) noexcept { return !(a > b); }
	[[nodiscard]] friend constexpr SF_INLINE bool operator<=(float a, SoftFloat b) noexcept { return !(a > b); }
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] friend constexpr SF_INLINE bool operator<=(SoftFloat a, int b) noexcept { return !(a > b); }
#endif
	[[nodiscard]] friend constexpr SF_INLINE bool operator<=(SoftFloat a, int32_t b) noexcept { return !(a > b); }
	[[nodiscard]] friend constexpr SF_INLINE bool operator<=(SoftFloat a, float b) noexcept { return !(a > b); }

	[[nodiscard]] friend constexpr SF_INLINE bool operator>=(SoftFloat a, SoftFloat b) noexcept { return !(a < b); }
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] friend constexpr SF_INLINE bool operator>=(int a, SoftFloat b) noexcept { return !(a < b); }
#endif
	[[nodiscard]] friend constexpr SF_INLINE bool operator>=(int32_t a, SoftFloat b) noexcept { return !(a < b); }
	[[nodiscard]] friend constexpr SF_INLINE bool operator>=(float a, SoftFloat b) noexcept { return !(a < b); }
#ifndef SF_INT_EQUALS_INT32
	[[nodiscard]] friend constexpr SF_INLINE bool operator>=(SoftFloat a, int b) noexcept { return !(a < b); }
#endif
	[[nodiscard]] friend constexpr SF_INLINE bool operator>=(SoftFloat a, int32_t b) noexcept { return !(a < b); }
	[[nodiscard]] friend constexpr SF_INLINE bool operator>=(SoftFloat a, float b) noexcept { return !(a < b); }

	// ------------------------------------------------------------------
	// Utility queries — constexpr
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr bool is_zero()     const noexcept { return mantissa == 0; }
	[[nodiscard]] constexpr bool is_negative() const noexcept { return mantissa < 0; }
	[[nodiscard]] constexpr bool is_positive() const noexcept { return mantissa > 0; }

	// ------------------------------------------------------------------
	// abs — constexpr, branch-free (ASR+EOR+SUB on ARM)
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SF_INLINE SoftFloat abs() const noexcept {
		return from_raw_unchecked(static_cast<int32_t>(abs32(mantissa)), exponent);
	}

	// ------------------------------------------------------------------
	// Clamp — constexpr
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SoftFloat clamp(SoftFloat lo, SoftFloat hi) const noexcept {
		if (*this < lo) return lo;
		if (*this > hi) return hi;
		return *this;
	}
	[[nodiscard]] constexpr SF_INLINE SoftFloat clamp(float lo, SoftFloat hi) const noexcept { return clamp(SoftFloat(lo), hi); }
	[[nodiscard]] constexpr SF_INLINE SoftFloat clamp(SoftFloat lo, float hi) const noexcept { return clamp(lo, SoftFloat(hi)); }
	[[nodiscard]] constexpr SF_INLINE SoftFloat clamp(float lo, float hi) const noexcept { return clamp(SoftFloat(lo), SoftFloat(hi)); }

	// ------------------------------------------------------------------
	// Math functions — constexpr via integer arithmetic only
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat sin() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat cos() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SinCosPair sincos() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat tan() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat asin() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat acos() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat atan() const noexcept
	{
		return atan2(*this, SoftFloat::one());
	}
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat sinh() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat cosh() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat tanh() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat inv_sqrt() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat sqrt()     const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat exp()     const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat log() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat log2() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat log10() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat pow(SoftFloat y) const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat trunc() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat floor() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat ceil() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat round() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat fract() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE IntFractPair modf() const noexcept;
	[[nodiscard]] constexpr SoftFloat copysign(SoftFloat sign) const noexcept;
	[[nodiscard]] constexpr SoftFloat fmod(SoftFloat y) const noexcept;
	[[nodiscard]] constexpr SoftFloat fma(SoftFloat b, SoftFloat c) const noexcept;
	friend constexpr SF_HOT SoftFloat atan2(SoftFloat y, SoftFloat x) noexcept;
	friend constexpr SF_HOT SoftFloat hypot(SoftFloat x, SoftFloat y) noexcept;
	friend constexpr SF_HOT SoftFloat lerp(SoftFloat a, SoftFloat b, SoftFloat t) noexcept;

	// ------------------------------------------------------------------
	// Compound assignment — constexpr
	// ------------------------------------------------------------------
	constexpr SoftFloat& operator+=(SoftFloat r) noexcept { *this = *this + r; return *this; }
	constexpr SoftFloat& operator-=(SoftFloat r) noexcept { *this = *this - r; return *this; }
	constexpr SoftFloat& operator*=(SoftFloat r) noexcept;
	constexpr SoftFloat& operator/=(SoftFloat r) noexcept { *this = *this / r; return *this; }

	// --- FMA autodetection: a += b * c  →  fused_mul_add(a, b, c) ---
	constexpr SF_HOT SF_INLINE SoftFloat& operator+=(const MulExpr& m) noexcept;
	constexpr SF_HOT SF_INLINE SoftFloat& operator-=(const MulExpr& m) noexcept;
	
	// ------------------------------------------------------------------
	// Constants
	// ------------------------------------------------------------------
	[[nodiscard]] static constexpr SoftFloat zero()     noexcept { return from_raw_unchecked(0, 0); }
	[[nodiscard]] static constexpr SoftFloat one()      noexcept { return from_raw_unchecked(MANT_MIN, -MANT_BITS); }
	[[nodiscard]] static constexpr SoftFloat neg_one()  noexcept { return from_raw_unchecked(-static_cast<int32_t>(MANT_MIN), -MANT_BITS); }
	[[nodiscard]] static constexpr SoftFloat half()     noexcept { return from_raw_unchecked(MANT_MIN, -MANT_BITS - 1); }
	[[nodiscard]] static constexpr SoftFloat two()      noexcept { return from_raw_unchecked(MANT_MIN, -MANT_BITS + 1); }
	[[nodiscard]] static constexpr SoftFloat three()    noexcept { return from_raw_unchecked(0x30000000, -28); }
	[[nodiscard]] static constexpr SoftFloat four()     noexcept { return from_raw_unchecked(MANT_MIN, -MANT_BITS + 2); }
	[[nodiscard]] static constexpr SoftFloat pi()       noexcept { return from_raw_unchecked(843314857, -28); }
	[[nodiscard]] static constexpr SoftFloat two_pi()   noexcept { return from_raw_unchecked(843314857, -27); }
	[[nodiscard]] static constexpr SoftFloat half_pi()  noexcept { return from_raw_unchecked(843314857, -29); }

	// ------------------------------------------------------------------
	// Fused operations (friend declarations)
	// ------------------------------------------------------------------
	friend constexpr SF_HOT SoftFloat fused_mul_add(SoftFloat a, SoftFloat b, SoftFloat c) noexcept;
	friend constexpr SF_HOT SoftFloat fused_mul_sub(SoftFloat a, SoftFloat b, SoftFloat c) noexcept;
	friend constexpr SF_HOT SoftFloat fused_mul_mul_add(SoftFloat a, SoftFloat b, SoftFloat c, SoftFloat d) noexcept;
	friend constexpr SF_HOT SoftFloat fused_mul_mul_sub(SoftFloat a, SoftFloat b, SoftFloat c, SoftFloat d) noexcept;
};

// =========================================================================
// Return types
// =========================================================================

struct IntFractPair { SoftFloat intpart; SoftFloat fracpart; };
struct SinCosPair   { SoftFloat sin; SoftFloat cos; };

// =========================================================================
// Expression-template proxy for a deferred single multiplication.
// Allows the compiler to fuse  a + b*c  into a single FMA call.
// =========================================================================
struct SoftFloat::MulExpr {
	SoftFloat lhs;
	SoftFloat rhs;

	[[nodiscard]] constexpr SF_INLINE SoftFloat eval() const noexcept {
		return SoftFloat::mul_plain(lhs, rhs);
	}

	[[nodiscard]] constexpr explicit operator float()     const noexcept { return eval().to_float(); }
	[[nodiscard]] constexpr float     to_float()          const noexcept { return eval().to_float(); }
	[[nodiscard]] constexpr int32_t   to_int32()          const noexcept { return eval().to_int32(); }
	[[nodiscard]] constexpr bool      is_zero()           const noexcept { return eval().is_zero(); }
	[[nodiscard]] constexpr bool      is_negative()       const noexcept { return eval().is_negative(); }
	[[nodiscard]] constexpr SoftFloat abs()               const noexcept { return eval().abs(); }
	[[nodiscard]] constexpr SoftFloat sqrt()              const noexcept { return eval().sqrt(); }
	[[nodiscard]] constexpr SoftFloat exp()               const noexcept { return eval().exp(); }
	[[nodiscard]] constexpr SoftFloat log()               const noexcept { return eval().log(); }
	[[nodiscard]] constexpr SoftFloat log2()              const noexcept { return eval().log2(); }
	[[nodiscard]] constexpr SoftFloat log10()             const noexcept { return eval().log10(); }
	[[nodiscard]] constexpr SoftFloat pow(SoftFloat y)    const noexcept { return eval().pow(y); }
	[[nodiscard]] constexpr SoftFloat trunc()             const noexcept { return eval().trunc(); }
	[[nodiscard]] constexpr SoftFloat floor()             const noexcept { return eval().floor(); }
	[[nodiscard]] constexpr SoftFloat ceil()              const noexcept { return eval().ceil(); }
	[[nodiscard]] constexpr SoftFloat round()             const noexcept { return eval().round(); }
	[[nodiscard]] constexpr SoftFloat fract()             const noexcept { return eval().fract(); }
	[[nodiscard]] constexpr IntFractPair modf()           const noexcept { return eval().modf(); }
	[[nodiscard]] constexpr SoftFloat copysign(SoftFloat sign) const noexcept { return eval().copysign(sign); }
	[[nodiscard]] constexpr SoftFloat fmod(SoftFloat y)        const noexcept { return eval().fmod(y); }
	[[nodiscard]] constexpr SoftFloat fma(SoftFloat b, SoftFloat c) const noexcept { return eval().fma(b, c); }
	[[nodiscard]] constexpr SoftFloat inv_sqrt()          const noexcept { return eval().inv_sqrt(); }
	[[nodiscard]] constexpr SoftFloat reciprocal()        const noexcept { return eval().reciprocal(); }
	[[nodiscard]] constexpr SoftFloat clamp(SoftFloat lo, SoftFloat hi) const noexcept {
		return eval().clamp(lo, hi);
	}
	[[nodiscard]] constexpr SoftFloat sin()               const noexcept { return eval().sin(); }
	[[nodiscard]] constexpr SoftFloat cos()               const noexcept { return eval().cos(); }
	[[nodiscard]] constexpr SinCosPair sincos()           const noexcept { return eval().sincos(); }
	[[nodiscard]] constexpr SoftFloat tan()               const noexcept { return eval().tan(); }
	[[nodiscard]] constexpr SoftFloat asin()              const noexcept { return eval().asin(); }
	[[nodiscard]] constexpr SoftFloat acos()              const noexcept { return eval().acos(); }
	[[nodiscard]] constexpr SoftFloat atan()              const noexcept { return eval().atan(); }
	[[nodiscard]] constexpr SoftFloat sinh()              const noexcept { return eval().sinh(); }
	[[nodiscard]] constexpr SoftFloat cosh()              const noexcept { return eval().cosh(); }
	[[nodiscard]] constexpr SoftFloat tanh()              const noexcept { return eval().tanh(); }
	[[nodiscard]] constexpr SoftFloat operator/(SoftFloat r)  const noexcept { return eval() / r; }
	[[nodiscard]] constexpr SoftFloat operator/(float r)      const noexcept { return eval() / SoftFloat(r); }
	[[nodiscard]] constexpr SoftFloat operator/(int32_t r)    const noexcept { return eval() / SoftFloat(r); }
	[[nodiscard]] constexpr SoftFloat operator>>(int s)       const noexcept { return eval() >> s; }
	[[nodiscard]] constexpr SoftFloat operator<<(int s)       const noexcept { return eval() << s; }

	// Negate the expression (flips lhs sign, lazy — no evaluation)
	[[nodiscard]] constexpr MulExpr operator-() const noexcept {
		MulExpr r = *this;
		r.lhs = -r.lhs;
		return r;
	}
};

// =========================================================================
// SoftFloat proxy constructor (defined here so MulExpr is complete)
// =========================================================================
constexpr SF_HOT SF_INLINE SoftFloat::SoftFloat(const MulExpr& m) noexcept {
	SoftFloat v = m.eval();
	mantissa = v.mantissa;
	exponent = v.exponent;
}

// =========================================================================
// operator* — returns deferred proxy (constexpr, no computation yet)
// =========================================================================
[[nodiscard]] constexpr SF_HOT SF_INLINE SoftFloat::MulExpr operator*(SoftFloat a, SoftFloat b) noexcept {
	return { a, b };
}
[[nodiscard]] constexpr SF_HOT SF_INLINE SoftFloat::MulExpr operator*(SoftFloat a, float b) noexcept {
	return a * SoftFloat(b);
}
#ifndef SF_INT_EQUALS_INT32
[[nodiscard]] constexpr SF_HOT SF_INLINE SoftFloat::MulExpr operator*(SoftFloat a, int b) noexcept {
	return a * SoftFloat(b);
}
#endif
[[nodiscard]] constexpr SF_HOT SF_INLINE SoftFloat::MulExpr operator*(SoftFloat a, int32_t b) noexcept {
	return a * SoftFloat(b);
}
[[nodiscard]] constexpr SF_HOT SF_INLINE SoftFloat::MulExpr operator*(float a, SoftFloat b) noexcept {
	return SoftFloat(a) * b;
}
#ifndef SF_INT_EQUALS_INT32
[[nodiscard]] constexpr SF_HOT SF_INLINE SoftFloat::MulExpr operator*(int a, SoftFloat b) noexcept {
	return SoftFloat(a) * b;
}
#endif
[[nodiscard]] constexpr SF_HOT SF_INLINE SoftFloat::MulExpr operator*(int32_t a, SoftFloat b) noexcept {
	return SoftFloat(a) * b;
}

// --- FMA autodetection: a += b * c  →  fused_mul_add(a, b, c) ---
constexpr SF_HOT SF_INLINE SoftFloat& SoftFloat::operator+=(const MulExpr& m) noexcept {
	*this = fused_mul_add(*this, m.lhs, m.rhs);
	return *this;
}
constexpr SF_HOT SF_INLINE SoftFloat& SoftFloat::operator-=(const MulExpr& m) noexcept {
	*this = fused_mul_sub(*this, m.lhs, m.rhs);
	return *this;
}
constexpr SoftFloat& SoftFloat::operator*=(SoftFloat r) noexcept {
	*this = *this * r;
	return *this;
}

// =========================================================================
// normalise_fast definition
// =========================================================================
constexpr SF_INLINE void SoftFloat::normalise_fast(int32_t& m, int32_t& e) noexcept
{
	if (SF_IS_CONSTEVAL()) {
		uint32_t sign = static_cast<uint32_t>(m >> 31);
		uint32_t a    = (static_cast<uint32_t>(m) ^ sign) - sign;
		if (LIKELY((a & MANT_TOP_TWO) == MANT_MIN)) {
			e = sat_exp(e);
			return;
		}
		if (UNLIKELY(a & MANT_OVERFLOW)) {
			a >>= 1;
			e += 1;
		}
		else {
			int lz = clz(a);
			int sh = lz - 2;
			e -= sh;
			a <<= sh;
		}
		e = sat_exp(e);
		m = static_cast<int32_t>((a ^ sign) - sign);
		return;
	}

#if defined(__arm__)
	{
		uint32_t sign = static_cast<uint32_t>(m >> 31);
		uint32_t a    = (static_cast<uint32_t>(m) ^ sign) - sign;

		if (LIKELY((a & MANT_TOP_TWO) == MANT_MIN)) {
			__asm__(
				"ssat %0, #8, %1"
				: "=r"(e)
				: "r"(e));
			return;
		}

		if (UNLIKELY(a & MANT_OVERFLOW)) {
			a >>= 1;
			e += 1;
		}
		else {
			uint32_t lz;
			__asm__("clz %0, %1" : "=r"(lz) : "r"(a));
			int32_t shift = static_cast<int32_t>(lz) - 2;
			// shift > 0 is guaranteed: bit 30 is clear (overflow handled above)
			// and bit 29 is clear (fast path handled above)
			a <<= shift;
			e -= shift;
		}

		__asm__("ssat %0, #8, %1" : "=r"(e) : "r"(e));
		m = static_cast<int32_t>((a ^ sign) - sign);
		return;
	}
#else
	{
		uint32_t sign = static_cast<uint32_t>(m >> 31);
		uint32_t a    = (static_cast<uint32_t>(m) ^ sign) - sign;
		if (LIKELY((a & MANT_TOP_TWO) == MANT_MIN)) {
			e = sat_exp(e);
			return;
		}
		if (UNLIKELY(a & MANT_OVERFLOW)) {
			a >>= 1;
			e += 1;
		}
		else {
			int lz = clz(a);
			int sh = lz - 2;
			e -= sh;
			a <<= sh;
		}
		e = sat_exp(e);
		m = static_cast<int32_t>((a ^ sign) - sign);
	}
#endif
}

// =========================================================================
// recip32 definition
// =========================================================================
[[nodiscard]] constexpr SF_CONST SF_INLINE uint32_t SoftFloat::recip32(uint32_t b) noexcept
{
	static constexpr uint32_t recip_tab[512] = {
		0x80000000u, 0x7FC01FF0u, 0x7F807F80u, 0x7F411E52u,
		0x7F01FC07u, 0x7EC31843u, 0x7E8472A8u, 0x7E460ADAu,
		0x7E07E07Eu, 0x7DC9F339u, 0x7D8C42B2u, 0x7D4ECE8Fu,
		0x7D119679u, 0x7CD49A16u, 0x7C97D910u, 0x7C5B5311u,
		0x7C1F07C1u, 0x7BE2F6CEu, 0x7BA71FE1u, 0x7B6B82A6u,
		0x7B301ECCu, 0x7AF4F3FEu, 0x7ABA01EAu, 0x7A7F4841u,
		0x7A44C6AFu, 0x7A0A7CE6u, 0x79D06A96u, 0x79968F6Fu,
		0x795CEB24u, 0x79237D65u, 0x78EA45E7u, 0x78B1445Cu,
		0x78787878u, 0x783FE1F0u, 0x78078078u, 0x77CF53C5u,
		0x77975B8Fu, 0x775F978Cu, 0x77280772u, 0x76F0AAF9u,
		0x76B981DAu, 0x76828BCEu, 0x764BC88Cu, 0x761537D0u,
		0x75DED952u, 0x75A8ACCFu, 0x7572B201u, 0x753CE8A4u,
		0x75075075u, 0x74D1E92Fu, 0x749CB28Fu, 0x7467AC55u,
		0x7432D63Du, 0x73FE3007u, 0x73C9B971u, 0x7395723Au,
		0x73615A24u, 0x732D70EDu, 0x72F9B658u, 0x72C62A24u,
		0x7292CC15u, 0x725F9BECu, 0x722C996Bu, 0x71F9C457u,
		0x71C71C71u, 0x7194A17Fu, 0x71625344u, 0x71303185u,
		0x70FE3C07u, 0x70CC728Fu, 0x709AD4E4u, 0x706962CCu,
		0x70381C0Eu, 0x70070070u, 0x6FD60FBAu, 0x6FA549B4u,
		0x6F74AE26u, 0x6F443CD9u, 0x6F13F596u, 0x6EE3D826u,
		0x6EB3E453u, 0x6E8419E6u, 0x6E5478ACu, 0x6E25006Eu,
		0x6DF5B0F7u, 0x6DC68A13u, 0x6D978B8Eu, 0x6D68B535u,
		0x6D3A06D3u, 0x6D0B8036u, 0x6CDD212Bu, 0x6CAEE97Fu,
		0x6C80D901u, 0x6C52EF7Fu, 0x6C252CC7u, 0x6BF790A8u,
		0x6BCA1AF2u, 0x6B9CCB74u, 0x6B6FA1FEu, 0x6B429E60u,
		0x6B15C06Bu, 0x6AE907EFu, 0x6ABC74BEu, 0x6A9006A9u,
		0x6A63BD81u, 0x6A37991Au, 0x6A0B9944u, 0x69DFBDD4u,
		0x69B4069Bu, 0x6988736Du, 0x695D041Du, 0x6931B880u,
		0x69069069u, 0x68DB8BACu, 0x68B0AA1Fu, 0x6885EB95u,
		0x685B4FE5u, 0x6830D6E4u, 0x68068068u, 0x67DC4C45u,
		0x67B23A54u, 0x67884A69u, 0x675E7C5Du, 0x6734D006u,
		0x670B453Bu, 0x66E1DBD4u, 0x66B893A9u, 0x668F6C91u,
		0x66666666u, 0x663D80FFu, 0x6614BC36u, 0x65EC17E3u,
		0x65C393E0u, 0x659B3006u, 0x6572EC2Fu, 0x654AC835u,
		0x6522C3F3u, 0x64FADF42u, 0x64D3199Eu, 0x64AB7401u,
		0x6483ED27u, 0x645C854Au, 0x64353C48u, 0x640E11FAu,
		0x63E7063Eu, 0x63C018F0u, 0x639949EBu, 0x6372990Eu,
		0x634C0634u, 0x6325913Cu, 0x62FF3A01u, 0x62D90062u,
		0x62B2E43Du, 0x628CE570u, 0x626703D8u, 0x62413F54u,
		0x621B97C2u, 0x61F60D02u, 0x61D09EF3u, 0x61AB4D72u,
		0x61861861u, 0x6160FF9Eu, 0x613C0309u, 0x61172283u,
		0x60F25DEAu, 0x60CDB520u, 0x60A92806u, 0x6084B67Au,
		0x60606060u, 0x603C2597u, 0x60180601u, 0x5FF4017Fu,
		0x5FD017F4u, 0x5FAC493Fu, 0x5F889545u, 0x5F64FBE6u,
		0x5F417D05u, 0x5F1E1885u, 0x5EFACE48u, 0x5ED79E31u,
		0x5EB48823u, 0x5E918C01u, 0x5E6EA9AEu, 0x5E4BE10Fu,
		0x5E293205u, 0x5E069C77u, 0x5DE42046u, 0x5DC1BD58u,
		0x5D9F7390u, 0x5D7D42D4u, 0x5D5B2B08u, 0x5D392C10u,
		0x5D1745D1u, 0x5CF57831u, 0x5CD3C315u, 0x5CB22661u,
		0x5C90A1FDu, 0x5C6F35CCu, 0x5C4DE1B6u, 0x5C2CA5A0u,
		0x5C0B8170u, 0x5BEA750Cu, 0x5BC9805Bu, 0x5BA8A344u,
		0x5B87DDADu, 0x5B672F7Cu, 0x5B46989Au, 0x5B2618ECu,
		0x5B05B05Bu, 0x5AE55ECDu, 0x5AC5242Au, 0x5AA5005Au,
		0x5A84F345u, 0x5A64FCD2u, 0x5A451CEAu, 0x5A255374u,
		0x5A05A05Au, 0x59E60382u, 0x59C67CD8u, 0x59A70C41u,
		0x5987B1A9u, 0x59686CF7u, 0x59493E14u, 0x592A24EBu,
		0x590B2164u, 0x58EC3368u, 0x58CD5AE2u, 0x58AE97BAu,
		0x588FE9DCu, 0x58715130u, 0x5852CDA0u, 0x58345F18u,
		0x58160581u, 0x57F7C0C5u, 0x57D990D0u, 0x57BB758Cu,
		0x579D6EE3u, 0x577F7CC0u, 0x57619F0Fu, 0x5743D5BBu,
		0x572620AEu, 0x57087FD4u, 0x56EAF319u, 0x56CD7A67u,
		0x56B015ACu, 0x5692C4D1u, 0x567587C4u, 0x56585E70u,
		0x563B48C2u, 0x561E46A4u, 0x56015805u, 0x55E47CD0u,
		0x55C7B4F1u, 0x55AB0055u, 0x558E5EE9u, 0x5571D09Au,
		0x55555555u, 0x5538ED06u, 0x551C979Au, 0x55005500u,
		0x54E42523u, 0x54C807F2u, 0x54ABFD5Au, 0x54900549u,
		0x54741FABu, 0x54584C70u, 0x543C8B84u, 0x5420DCD6u,
		0x54054054u, 0x53E9B5EBu, 0x53CE3D8Bu, 0x53B2D721u,
		0x5397829Cu, 0x537C3FEBu, 0x53610EFBu, 0x5345EFBCu,
		0x532AE21Cu, 0x530FE60Bu, 0x52F4FB76u, 0x52DA224Eu,
		0x52BF5A81u, 0x52A4A3FEu, 0x5289FEB5u, 0x526F6A96u,
		0x5254E78Eu, 0x523A758Fu, 0x52201488u, 0x5205C467u,
		0x51EB851Eu, 0x51D1569Cu, 0x51B738D1u, 0x519D2BADu,
		0x51832F1Fu, 0x51694319u, 0x514F678Bu, 0x51359C64u,
		0x511BE195u, 0x5102370Fu, 0x50E89CC2u, 0x50CF129Fu,
		0x50B59897u, 0x509C2E9Au, 0x5082D499u, 0x50698A85u,
		0x50505050u, 0x503725EAu, 0x501E0B44u, 0x50050050u,
		0x4FEC04FEu, 0x4FD31941u, 0x4FBA3D0Au, 0x4FA1704Au,
		0x4F88B2F3u, 0x4F7004F7u, 0x4F576646u, 0x4F3ED6D4u,
		0x4F265691u, 0x4F0DE571u, 0x4EF58364u, 0x4EDD305Du,
		0x4EC4EC4Eu, 0x4EACB72Au, 0x4E9490E1u, 0x4E7C7968u,
		0x4E6470B0u, 0x4E4C76ABu, 0x4E348B4Du, 0x4E1CAE88u,
		0x4E04E04Eu, 0x4DED2092u, 0x4DD56F47u, 0x4DBDCC5Fu,
		0x4DA637CFu, 0x4D8EB188u, 0x4D77397Eu, 0x4D5FCFA4u,
		0x4D4873ECu, 0x4D31264Bu, 0x4D19E6B3u, 0x4D02B518u,
		0x4CEB916Du, 0x4CD47BA5u, 0x4CBD73B5u, 0x4CA67990u,
		0x4C8F8D28u, 0x4C78AE73u, 0x4C61DD63u, 0x4C4B19EDu,
		0x4C346404u, 0x4C1DBB9Du, 0x4C0720ABu, 0x4BF09322u,
		0x4BDA12F6u, 0x4BC3A01Cu, 0x4BAD3A87u, 0x4B96E22Du,
		0x4B809701u, 0x4B6A58F7u, 0x4B542804u, 0x4B3E041Du,
		0x4B27ED36u, 0x4B11E343u, 0x4AFBE639u, 0x4AE5F60Du,
		0x4AD012B4u, 0x4ABA3C21u, 0x4AA4724Bu, 0x4A8EB526u,
		0x4A7904A7u, 0x4A6360C3u, 0x4A4DC96Eu, 0x4A383E9Fu,
		0x4A22C04Au, 0x4A0D4E64u, 0x49F7E8E2u, 0x49E28FBAu,
		0x49CD42E2u, 0x49B8024Du, 0x49A2CDF3u, 0x498DA5C8u,
		0x497889C2u, 0x496379D6u, 0x494E75FAu, 0x49397E24u,
		0x49249249u, 0x490FB25Fu, 0x48FADE5Cu, 0x48E61636u,
		0x48D159E2u, 0x48BCA957u, 0x48A8048Au, 0x48936B72u,
		0x487EDE04u, 0x486A5C37u, 0x4855E601u, 0x48417B57u,
		0x482D1C31u, 0x4818C884u, 0x48048048u, 0x47F04371u,
		0x47DC11F7u, 0x47C7EBCFu, 0x47B3D0F1u, 0x479FC154u,
		0x478BBCECu, 0x4777C3B2u, 0x4763D59Cu, 0x474FF2A1u,
		0x473C1AB6u, 0x47284DD4u, 0x47148BF0u, 0x4700D502u,
		0x46ED2901u, 0x46D987E3u, 0x46C5F19Fu, 0x46B2662Du,
		0x469EE584u, 0x468B6F9Au, 0x46780467u, 0x4664A3E2u,
		0x46514E02u, 0x463E02BEu, 0x462AC20Eu, 0x46178BE9u,
		0x46046046u, 0x45F13F1Cu, 0x45DE2864u, 0x45CB1C14u,
		0x45B81A25u, 0x45A5228Cu, 0x45923543u, 0x457F5241u,
		0x456C797Du, 0x4559AAF0u, 0x4546E68Fu, 0x45342C55u,
		0x45217C38u, 0x450ED630u, 0x44FC3A34u, 0x44E9A83Eu,
		0x44D72044u, 0x44C4A23Fu, 0x44B22E27u, 0x449FC3F4u,
		0x448D639Du, 0x447B0D1Bu, 0x4468C066u, 0x44567D76u,
		0x44444444u, 0x443214C7u, 0x441FEEF8u, 0x440DD2CEu,
		0x43FBC043u, 0x43E9B74Fu, 0x43D7B7EAu, 0x43C5C20Du,
		0x43B3D5AFu, 0x43A1F2CAu, 0x43901956u, 0x437E494Bu,
		0x436C82A2u, 0x435AC553u, 0x43491158u, 0x433766A9u,
		0x4325C53Eu, 0x43142D11u, 0x43029E1Au, 0x42F11851u,
		0x42DF9BB0u, 0x42CE2830u, 0x42BCBDC8u, 0x42AB5C73u,
		0x429A0429u, 0x4288B4E3u, 0x42776E9Au, 0x42663147u,
		0x4254FCE4u, 0x4243D168u, 0x4232AECDu, 0x4221950Du,
		0x42108421u, 0x41FF7C01u, 0x41EE7CA6u, 0x41DD860Bu,
		0x41CC9829u, 0x41BBB2F8u, 0x41AAD671u, 0x419A0290u,
		0x4189374Bu, 0x4178749Eu, 0x4167BA81u, 0x415708EEu,
		0x41465FDFu, 0x4135BF4Cu, 0x41252730u, 0x41149783u,
		0x41041041u, 0x40F39161u, 0x40E31ADEu, 0x40D2ACB1u,
		0x40C246D4u, 0x40B1E941u, 0x40A193F1u, 0x409146DFu,
		0x40810204u, 0x4070C559u, 0x406090D9u, 0x4050647Du,
		0x40404040u, 0x4030241Bu, 0x40201008u, 0x40100401u,
	};

	uint32_t b2 = b << 1;
	uint32_t idx = (b2 >> 21) & 0x1FFu;
	uint32_t Y = recip_tab[idx];

#if defined(__arm__)
	if (!SF_IS_CONSTEVAL()) {
		uint32_t lo, hi;

		__asm__(
			"umull %0, %1, %2, %3"
			: "=&r"(lo),
			  "=&r"(hi)
			: "r"(b2),
			  "r"(Y));

		uint32_t q30 = (hi << 2) | (lo >> 30);
		uint32_t r30 = lo & 0x3FFFFFFFu;
		int32_t  err = static_cast<int32_t>(0x80000000u - q30 - (r30 ? 1u : 0u));

		uint32_t abs_err = static_cast<uint32_t>(err < 0 ? -err : err);
		__asm__(
			"umull %0, %1, %2, %3"
			: "=&r"(lo),
			  "=&r"(hi)
			: "r"(Y),
			  "r"(abs_err));
		uint32_t dY_mag = (hi << 1) | (lo >> 31);
		int32_t  dY = (err < 0) ? -static_cast<int32_t>(dY_mag)
		                        :  static_cast<int32_t>(dY_mag);

		uint32_t result = Y;
		if (dY < 0) result -= static_cast<uint32_t>(-dY);
		else        result += static_cast<uint32_t>( dY);
		return result;
	}
#endif

	uint64_t bY = static_cast<uint64_t>(b2) * Y;
	int64_t  err64 = static_cast<int64_t>(1ULL << 61) - static_cast<int64_t>(bY);
	int32_t  err = static_cast<int32_t>(err64 >> 30);
	int64_t  dY64 = (static_cast<int64_t>(Y) * err) >> 31;
	return static_cast<uint32_t>(static_cast<int64_t>(Y) + dY64);
}


// =========================================================================
// Fused arithmetic — constexpr (implicitly inline since constexpr)
// =========================================================================

[[nodiscard]] constexpr SF_HOT SoftFloat fused_mul_add(SoftFloat a, SoftFloat b, SoftFloat c) noexcept {
	if (UNLIKELY(!b.mantissa || !c.mantissa)) return a;
	if (UNLIKELY(!a.mantissa)) return SoftFloat::mul_plain(b, c);

#ifdef __arm__
	int32_t pm, pe;
	if (SF_IS_CONSTEVAL()) {
		int64_t prod = static_cast<int64_t>(b.mantissa) * static_cast<int64_t>(c.mantissa);
		pm = static_cast<int32_t>(prod >> 29);
		pe = b.exponent + c.exponent + 29;
	}
	else {
		int32_t lo, hi;
		__asm__("smull %0, %1, %2, %3"
			: "=&r"(lo), "=&r"(hi)
			: "r"(b.mantissa), "r"(c.mantissa));
		pm = (hi << 3) | (static_cast<uint32_t>(lo) >> 29);
		pe = b.exponent + c.exponent + 29;
	}
#else
	int32_t pm, pe;
	int64_t prod = static_cast<int64_t>(b.mantissa) * static_cast<int64_t>(c.mantissa);
	pm = static_cast<int32_t>(prod >> 29);
	pe = b.exponent + c.exponent + 29;
#endif

	uint32_t norm = static_cast<uint32_t>(pm ^ (pm >> 31)) >> 30;
	pm >>= norm;
	pe += static_cast<int32_t>(norm);

	int d = a.exponent - pe;
	if (d >= 31) return a;
	if (d <= -31) {
		if (UNLIKELY(pe > SoftFloat::EXP_MAX))
			return SoftFloat::from_raw_unchecked(pm >= 0 ? SoftFloat::MANT_MIN : -static_cast<int32_t>(SoftFloat::MANT_MIN), SoftFloat::EXP_MAX);
		if (UNLIKELY(pe < SoftFloat::EXP_MIN))
			return SoftFloat::zero();
		return SoftFloat::from_raw_unchecked(pm, pe);
	}

	int32_t am = a.mantissa;
	if (d == 0) return SoftFloat::finish_addsub(am + pm, pe);
	if (d > 0) {
		pm >>= d;
		return SoftFloat::finish_addsub(am + pm, a.exponent);
	}
	am >>= -d;
	return SoftFloat::finish_addsub(am + pm, SoftFloat::sat_exp(pe));
}

[[nodiscard]] constexpr SF_HOT SoftFloat fused_mul_sub(SoftFloat a, SoftFloat b, SoftFloat c) noexcept {
	if (UNLIKELY(!b.mantissa || !c.mantissa)) return a;
	if (UNLIKELY(!a.mantissa)) return -SoftFloat::mul_plain(b, c);

	int32_t pm, pe;
#ifdef __arm__
	if (SF_IS_CONSTEVAL()) {
		int64_t prod = static_cast<int64_t>(b.mantissa) * static_cast<int64_t>(c.mantissa);
		pm = static_cast<int32_t>(prod >> 29);
		pe = b.exponent + c.exponent + 29;
	}
	else {
		int32_t lo, hi;
		__asm__("smull %0, %1, %2, %3"
			: "=&r"(lo), "=&r"(hi)
			: "r"(b.mantissa), "r"(c.mantissa));
		pm = (hi << 3) | (static_cast<uint32_t>(lo) >> 29);
		pe = b.exponent + c.exponent + 29;
	}
#else
	int64_t prod = static_cast<int64_t>(b.mantissa) * static_cast<int64_t>(c.mantissa);
	pm = static_cast<int32_t>(prod >> 29);
	pe = b.exponent + c.exponent + 29;
#endif

	uint32_t norm = static_cast<uint32_t>(pm ^ (pm >> 31)) >> 30;
	pm >>= norm;
	pe += static_cast<int32_t>(norm);
	pm = -pm;

	int d = a.exponent - pe;
	if (d >= 31) return a;
	if (d <= -31) {
		if (UNLIKELY(pe > SoftFloat::EXP_MAX))
			return SoftFloat::from_raw_unchecked(pm >= 0 ? SoftFloat::MANT_MIN : -static_cast<int32_t>(SoftFloat::MANT_MIN), SoftFloat::EXP_MAX);
		if (UNLIKELY(pe < SoftFloat::EXP_MIN))
			return SoftFloat::zero();
		return SoftFloat::from_raw_unchecked(pm, pe);
	}

	int32_t am = a.mantissa;
	if (d == 0) return SoftFloat::finish_addsub(am + pm, pe);
	if (d > 0) {
		pm >>= d;
		return SoftFloat::finish_addsub(am + pm, a.exponent);
	}
	am >>= -d;
	return SoftFloat::finish_addsub(am + pm, SoftFloat::sat_exp(pe));
}

[[nodiscard]] constexpr SF_HOT SoftFloat fused_mul_mul_add(SoftFloat a, SoftFloat b, SoftFloat c, SoftFloat d) noexcept {
	bool abz = (!a.mantissa || !b.mantissa);
	bool cdz = (!c.mantissa || !d.mantissa);
	if (UNLIKELY(abz || cdz)) {
		if (abz && cdz) return SoftFloat::zero();
		if (abz)        return SoftFloat::mul_plain(c, d);
		return SoftFloat::mul_plain(a, b);
	}

	int32_t pm1, pe1, pm2, pe2;
#ifdef __arm__
	if (SF_IS_CONSTEVAL()) {
		int64_t p1 = static_cast<int64_t>(a.mantissa) * static_cast<int64_t>(b.mantissa);
		pm1 = static_cast<int32_t>(p1 >> 29);
		pe1 = a.exponent + b.exponent + 29;
		int64_t p2 = static_cast<int64_t>(c.mantissa) * static_cast<int64_t>(d.mantissa);
		pm2 = static_cast<int32_t>(p2 >> 29);
		pe2 = c.exponent + d.exponent + 29;
	}
	else {
		int32_t lo1, hi1;
		__asm__("smull %0, %1, %2, %3"
			: "=&r"(lo1), "=&r"(hi1)
			: "r"(a.mantissa), "r"(b.mantissa));
		pm1 = (hi1 << 3) | (static_cast<uint32_t>(lo1) >> 29);
		pe1 = a.exponent + b.exponent + 29;

		int32_t lo2, hi2;
		__asm__("smull %0, %1, %2, %3"
			: "=&r"(lo2), "=&r"(hi2)
			: "r"(c.mantissa), "r"(d.mantissa));
		pm2 = (hi2 << 3) | (static_cast<uint32_t>(lo2) >> 29);
		pe2 = c.exponent + d.exponent + 29;
	}
#else
	int64_t p1 = static_cast<int64_t>(a.mantissa) * static_cast<int64_t>(b.mantissa);
	pm1 = static_cast<int32_t>(p1 >> 29);
	pe1 = a.exponent + b.exponent + 29;
	int64_t p2 = static_cast<int64_t>(c.mantissa) * static_cast<int64_t>(d.mantissa);
	pm2 = static_cast<int32_t>(p2 >> 29);
	pe2 = c.exponent + d.exponent + 29;
#endif

	uint32_t n1 = static_cast<uint32_t>(pm1 ^ (pm1 >> 31)) >> 30;
	pm1 >>= n1; pe1 += n1;
	uint32_t n2 = static_cast<uint32_t>(pm2 ^ (pm2 >> 31)) >> 30;
	pm2 >>= n2; pe2 += n2;

	int d_exp = pe1 - pe2;
	if (d_exp >= 31) {
		if (UNLIKELY(pe1 > SoftFloat::EXP_MAX)) return SoftFloat::from_raw_unchecked(pm1 >= 0 ? SoftFloat::MANT_MIN : -static_cast<int32_t>(SoftFloat::MANT_MIN), SoftFloat::EXP_MAX);
		if (UNLIKELY(pe1 < SoftFloat::EXP_MIN)) return SoftFloat::zero();
		return SoftFloat::from_raw_unchecked(pm1, pe1);
	}
	if (d_exp <= -31) {
		if (UNLIKELY(pe2 > SoftFloat::EXP_MAX)) return SoftFloat::from_raw_unchecked(pm2 >= 0 ? SoftFloat::MANT_MIN : -static_cast<int32_t>(SoftFloat::MANT_MIN), SoftFloat::EXP_MAX);
		if (UNLIKELY(pe2 < SoftFloat::EXP_MIN)) return SoftFloat::zero();
		return SoftFloat::from_raw_unchecked(pm2, pe2);
	}

	if (d_exp == 0) return SoftFloat::finish_addsub(pm1 + pm2, pe1);

	int32_t exp;
	if (d_exp > 0) { pm2 >>= d_exp; exp = pe1; }
	else { pm1 >>= -d_exp; exp = pe2; }

	if (UNLIKELY(exp > SoftFloat::EXP_MAX || exp < SoftFloat::EXP_MIN)) {
		int32_t s = pm1 + pm2;
		if (UNLIKELY(s == 0)) return SoftFloat::zero();
		SoftFloat::normalise_fast(s, exp);
		return SoftFloat::from_raw_unchecked(s, exp);
	}
	return SoftFloat::finish_addsub(pm1 + pm2, exp);
}

[[nodiscard]] constexpr SF_HOT SoftFloat fused_mul_mul_sub(SoftFloat a, SoftFloat b,
	SoftFloat c, SoftFloat d) noexcept {
	return fused_mul_mul_add(a, b, c, SoftFloat::from_raw_unchecked(-d.mantissa, d.exponent));
}

// =========================================================================
// Mixed expression-template operators
// =========================================================================

[[nodiscard]] constexpr SF_INLINE
SoftFloat operator+(const SoftFloat::MulExpr& x, const SoftFloat::MulExpr& y) noexcept {
	return fused_mul_mul_add(x.lhs, x.rhs, y.lhs, y.rhs);
}
[[nodiscard]] constexpr SF_INLINE
SoftFloat operator-(const SoftFloat::MulExpr& x, const SoftFloat::MulExpr& y) noexcept {
	return fused_mul_mul_sub(x.lhs, x.rhs, y.lhs, y.rhs);
}

[[nodiscard]] constexpr SF_INLINE
SoftFloat operator+(SoftFloat a, const SoftFloat::MulExpr& m) noexcept {
	return fused_mul_add(a, m.lhs, m.rhs);
}
[[nodiscard]] constexpr SF_INLINE
SoftFloat operator-(SoftFloat a, const SoftFloat::MulExpr& m) noexcept {
	return fused_mul_sub(a, m.lhs, m.rhs);
}

[[nodiscard]] constexpr SF_INLINE
SoftFloat operator+(const SoftFloat::MulExpr& m, SoftFloat a) noexcept {
	return fused_mul_add(a, m.lhs, m.rhs);
}

[[nodiscard]] constexpr SF_INLINE
SoftFloat operator-(const SoftFloat::MulExpr& m, SoftFloat a) noexcept {
	return -fused_mul_sub(a, m.lhs, m.rhs);
}

// =========================================================================
// User-defined literals — constexpr
// =========================================================================
[[nodiscard]] constexpr SoftFloat operator""_sf(long double v) noexcept {
	return SoftFloat(static_cast<float>(v));
}
[[nodiscard]] constexpr SoftFloat operator""_sf(unsigned long long v) noexcept {
	return SoftFloat(static_cast<int32_t>(v));
}

// =========================================================================
// Convenience free functions — constexpr, no sf_ prefix
// =========================================================================
[[nodiscard]] constexpr SoftFloat abs(SoftFloat x)                                noexcept { return x.abs(); }
[[nodiscard]] constexpr SoftFloat sqrt(SoftFloat x)                               noexcept { return x.sqrt(); }
[[nodiscard]] constexpr SoftFloat exp(SoftFloat x)                                noexcept { return x.exp(); }
[[nodiscard]] constexpr SoftFloat log(SoftFloat x)                                noexcept { return x.log(); }
[[nodiscard]] constexpr SoftFloat log2(SoftFloat x)                               noexcept { return x.log2(); }
[[nodiscard]] constexpr SoftFloat log10(SoftFloat x)                              noexcept { return x.log10(); }
[[nodiscard]] constexpr SoftFloat pow(SoftFloat x, SoftFloat y)                   noexcept { return x.pow(y); }
[[nodiscard]] constexpr SoftFloat trunc(SoftFloat x)                              noexcept { return x.trunc(); }
[[nodiscard]] constexpr SoftFloat floor(SoftFloat x)                              noexcept { return x.floor(); }
[[nodiscard]] constexpr SoftFloat ceil(SoftFloat x)                               noexcept { return x.ceil(); }
[[nodiscard]] constexpr SoftFloat round(SoftFloat x)                              noexcept { return x.round(); }
[[nodiscard]] constexpr SoftFloat fract(SoftFloat x)                              noexcept { return x.fract(); }
[[nodiscard]] constexpr IntFractPair modf(SoftFloat x)                            noexcept { return x.modf(); }
[[nodiscard]] constexpr SoftFloat copysign(SoftFloat x, SoftFloat sign)           noexcept { return x.copysign(sign); }
[[nodiscard]] constexpr SoftFloat fmod(SoftFloat x, SoftFloat y)                  noexcept { return x.fmod(y); }
[[nodiscard]] constexpr SoftFloat fma(SoftFloat x, SoftFloat b, SoftFloat c)      noexcept { return x.fma(b, c); }
[[nodiscard]] constexpr SoftFloat inv_sqrt(SoftFloat x)                           noexcept { return x.inv_sqrt(); }
[[nodiscard]] constexpr SoftFloat reciprocal(SoftFloat x)                         noexcept { return x.reciprocal(); }
[[nodiscard]] constexpr SoftFloat min(SoftFloat a, SoftFloat b)                   noexcept { return (a < b) ? a : b; }
[[nodiscard]] constexpr SoftFloat max(SoftFloat a, SoftFloat b)                   noexcept { return (a > b) ? a : b; }
[[nodiscard]] constexpr SoftFloat clamp(SoftFloat v, SoftFloat lo, SoftFloat hi)  noexcept { return v.clamp(lo, hi); }
[[nodiscard]] constexpr SinCosPair sincos(SoftFloat x)                            noexcept { return x.sincos(); }
[[nodiscard]] constexpr SoftFloat sin(SoftFloat x)                                noexcept { return x.sin(); }
[[nodiscard]] constexpr SoftFloat cos(SoftFloat x)                                noexcept { return x.cos(); }
[[nodiscard]] constexpr SoftFloat tan(SoftFloat x)                                noexcept { return x.tan(); }
[[nodiscard]] constexpr SoftFloat asin(SoftFloat x)                               noexcept { return x.asin(); }
[[nodiscard]] constexpr SoftFloat acos(SoftFloat x)                               noexcept { return x.acos(); }
[[nodiscard]] constexpr SoftFloat atan(SoftFloat x)                               noexcept { return x.atan(); }
[[nodiscard]] constexpr SoftFloat sinh(SoftFloat x)                               noexcept { return x.sinh(); }
[[nodiscard]] constexpr SoftFloat cosh(SoftFloat x)                               noexcept { return x.cosh(); }
[[nodiscard]] constexpr SoftFloat tanh(SoftFloat x)                               noexcept { return x.tanh(); }

constexpr SF_HOT SinCosPair SoftFloat::sincos() const noexcept
{
	static constexpr int32_t SF_SIN_Q30[257] = {
		         0,   26350943,   52686014,   78989349,
		 105245103,  131437462,  157550647,  183568930,
		 209476638,  235258165,  260897982,  286380643,
		 311690799,  336813204,  361732726,  386434353,
		 410903207,  435124548,  459083786,  482766489,
		 506158392,  529245404,  552013618,  574449320,
		 596538995,  618269338,  639627258,  660599890,
		 681174602,  701339000,  721080937,  740388522,
		 759250125,  777654384,  795590213,  813046808,
		 830013654,  846480531,  862437520,  877875009,
		 892783698,  907154608,  920979082,  934248793,
		 946955747,  959092290,  970651112,  981625251,
		 992008094, 1001793390, 1010975242, 1019548121,
		1027506862, 1034846671, 1041563127, 1047652185,
		1053110176, 1057933813, 1062120190, 1065666786,
		1068571464, 1070832474, 1072448455, 1073418433,
		1073741824, 1073418433, 1072448455, 1070832474,
		1068571464, 1065666786, 1062120190, 1057933813,
		1053110176, 1047652185, 1041563127, 1034846671,
		1027506862, 1019548121, 1010975242, 1001793390,
		 992008094,  981625251,  970651112,  959092290,
		 946955747,  934248793,  920979082,  907154608,
		 892783698,  877875009,  862437520,  846480531,
		 830013654,  813046808,  795590213,  777654384,
		 759250125,  740388522,  721080937,  701339000,
		 681174602,  660599890,  639627258,  618269338,
		 596538995,  574449320,  552013618,  529245404,
		 506158392,  482766489,  459083786,  435124548,
		 410903207,  386434353,  361732726,  336813204,
		 311690799,  286380643,  260897982,  235258165,
		 209476638,  183568930,  157550647,  131437462,
		 105245103,   78989349,   52686014,   26350943,
		         0,  -26350943,  -52686014,  -78989349,
		-105245103, -131437462, -157550647, -183568930,
		-209476638, -235258165, -260897982, -286380643,
		-311690799, -336813204, -361732726, -386434353,
		-410903207, -435124548, -459083786, -482766489,
		-506158392, -529245404, -552013618, -574449320,
		-596538995, -618269338, -639627258, -660599890,
		-681174602, -701339000, -721080937, -740388522,
		-759250125, -777654384, -795590213, -813046808,
		-830013654, -846480531, -862437520, -877875009,
		-892783698, -907154608, -920979082, -934248793,
		-946955747, -959092290, -970651112, -981625251,
		-992008094,-1001793390,-1010975242,-1019548121,
	   -1027506862,-1034846671,-1041563127,-1047652185,
	   -1053110176,-1057933813,-1062120190,-1065666786,
	   -1068571464,-1070832474,-1072448455,-1073418433,
	   -1073741824,-1073418433,-1072448455,-1070832474,
	   -1068571464,-1065666786,-1062120190,-1057933813,
	   -1053110176,-1047652185,-1041563127,-1034846671,
	   -1027506862,-1019548121,-1010975242,-1001793390,
		-992008094, -981625251, -970651112, -959092290,
		-946955747, -934248793, -920979082, -907154608,
		-892783698, -877875009, -862437520, -846480531,
		-830013654, -813046808, -795590213, -777654384,
		-759250125, -740388522, -721080937, -701339000,
		-681174602, -660599890, -639627258, -618269338,
		-596538995, -574449320, -552013618, -529245404,
		-506158392, -482766489, -459083786, -435124548,
		-410903207, -386434353, -361732726, -336813204,
		-311690799, -286380643, -260897982, -235258165,
		-209476638, -183568930, -157550647, -131437462,
		-105245103,  -78989349,  -52686014,  -26350943,
		         0,
	};

	if (UNLIKELY(mantissa == 0))
		return { zero(), one() };

	int32_t m = mantissa;
	int32_t e = exponent;

	if (e == -27) {
		const int32_t tpm = 843314857;
		if (m >= tpm) {
			m -= tpm;
		}
		else if (m < 0) {
			m += tpm;
			if (m < 0) m += tpm;
		}
	}
	else if (e == -26) {
		const uint32_t tpm = 843314857;
		uint32_t am = abs32(m);
		uint32_t M  = am << 1;
		uint32_t k, rem;
#if defined(__arm__)
		if (!SF_IS_CONSTEVAL()) {
			__asm__(
				"udiv %0, %1, %2"
				: "=r"(k)
				: "r"(M), "r"(tpm));
		}
		else
#endif
			k = M / tpm;

		rem = M - k * tpm;
		if (m < 0 && rem != 0)
			m = static_cast<int32_t>(tpm - rem);
		else
			m = static_cast<int32_t>(rem);
		e = -27;
	}
	else if (e == -25) {
		const uint32_t tpm = 843314857;
		uint32_t am = abs32(m);
		uint32_t M  = am << 2;
		uint32_t k, rem;
#if defined(__arm__)
		if (!SF_IS_CONSTEVAL()) {
			__asm__(
				"udiv %0, %1, %2"
				: "=r"(k)
				: "r"(M), "r"(tpm));
		}
		else
#endif
			k = M / tpm;

		rem = M - k * tpm;
		if (m < 0 && rem != 0)
			m = static_cast<int32_t>(tpm - rem);
		else
			m = static_cast<int32_t>(rem);
		e = -27;
	}
	else if (e > -25) {
		SoftFloat xi = SoftFloat::from_raw_unchecked(mantissa, exponent);
		constexpr int32_t INV_2PI_M = 683565276;
		constexpr int32_t INV_2PI_E = -32;
		constexpr int32_t TWO_PI_M  = 843314857;
		constexpr int32_t TWO_PI_E  = -27;

		int32_t ki = (xi * SoftFloat::from_raw_unchecked(INV_2PI_M, INV_2PI_E)).to_int32();
		if (ki != 0)
			xi = xi - SoftFloat(ki) * SoftFloat::from_raw_unchecked(TWO_PI_M, TWO_PI_E);

		const SoftFloat two_pi = SoftFloat::from_raw_unchecked(TWO_PI_M, TWO_PI_E);
		if (xi.mantissa < 0) xi = xi + two_pi;
		if (!(xi < two_pi))  xi = xi - two_pi;

		m = xi.mantissa;
		e = xi.exponent;
	}

	const bool neg = m < 0;
	const uint32_t am = abs32(m);

	constexpr int32_t K_Q25 = 1367130551;

	uint32_t phase;
#if defined(__arm__)
	if (!SF_IS_CONSTEVAL()) {
		int32_t lo, hi;
		__asm__(
			"umull %0, %1, %2, %3"
			: "=&r"(lo),
			  "=&r"(hi)
			: "r"(am),
			  "r"(K_Q25));

		const int32_t s = 1 - e;
		if (s < 32) {
			phase = (static_cast<uint32_t>(hi) << (32 - s))
			      | (static_cast<uint32_t>(lo) >> s);
		}
		else if (s < 64) {
			phase = static_cast<uint32_t>(hi) >> (s - 32);
		}
		else {
			phase = 0;
		}
	}
	else
#endif
	{
		uint64_t prod = static_cast<uint64_t>(am) * static_cast<uint64_t>(K_Q25);
		int32_t s = 1 - e;
		phase = (s < 64) ? static_cast<uint32_t>(prod >> s) : 0u;
	}

	const uint32_t idx  = (phase >> 24) & 0xFFu;
	const uint32_t frac = phase & 0xFFFFFFu;

	const uint32_t s_idx0 = idx;
	const uint32_t s_idx1 = idx + 1u;
	const uint32_t c_idx0 = (idx + 64u) & 0xFFu;
	const uint32_t c_idx1 = c_idx0 + 1u;

	const int32_t s0 = SF_SIN_Q30[s_idx0];
	const int32_t s1 = SF_SIN_Q30[s_idx1];
	const int32_t c0 = SF_SIN_Q30[c_idx0];
	const int32_t c1 = SF_SIN_Q30[c_idx1];

	int32_t sin_q30, cos_q30;
#if defined(__arm__)
	if (!SF_IS_CONSTEVAL()) {
		int32_t ds = s1 - s0;
		int32_t dc = c1 - c0;
		int32_t lo_s, hi_s, lo_c, hi_c;

		__asm__(
			"smull %0, %1, %2, %3"
			: "=&r"(lo_s),
			  "=&r"(hi_s)
			: "r"(ds),
			  "r"(static_cast<int32_t>(frac)));
		__asm__(
			"smull %0, %1, %2, %3"
			: "=&r"(lo_c),
			  "=&r"(hi_c)
			: "r"(dc),
			  "r"(static_cast<int32_t>(frac)));

		sin_q30 = s0 + ((hi_s << 8) | (static_cast<uint32_t>(lo_s) >> 24));
		cos_q30 = c0 + ((hi_c << 8) | (static_cast<uint32_t>(lo_c) >> 24));
	}
	else
#endif
	{
		int64_t ps = static_cast<int64_t>(s1 - s0) * static_cast<int64_t>(frac);
		int64_t pc = static_cast<int64_t>(c1 - c0) * static_cast<int64_t>(frac);
		sin_q30 = s0 + static_cast<int32_t>(ps >> 24);
		cos_q30 = c0 + static_cast<int32_t>(pc >> 24);
	}

	if (neg) sin_q30 = -sin_q30;

	auto from_q30 = [](int32_t q) -> SoftFloat {
		if (q == 0) return SoftFloat::zero();
		uint32_t a = SoftFloat::abs32(q);
		if (a >= SoftFloat::MANT_OVERFLOW) {
			q >>= 1;
			return SoftFloat::from_raw_unchecked(q, -29);
		}
		if (a >= SoftFloat::MANT_MIN)
			return SoftFloat::from_raw_unchecked(q, -30);
		int shift = SoftFloat::clz(a) - 2;
		q <<= shift;
		return SoftFloat::from_raw_unchecked(q, -30 - shift);
	};

	return { from_q30(sin_q30), from_q30(cos_q30) };
}

constexpr SF_HOT SoftFloat SoftFloat::tan() const noexcept {
	if (UNLIKELY(mantissa == 0)) return zero();

	auto [s, c] = sincos();
	if (UNLIKELY(c.mantissa == 0))
		return from_raw_unchecked(s.mantissa >= 0 ? MANT_MIN : -static_cast<int32_t>(MANT_MIN), EXP_MAX);

	SoftFloat c_inv = c.reciprocal();

	int32_t rm, re = s.exponent + c_inv.exponent + 29;
	if (SF_IS_CONSTEVAL()) {
		int64_t prod = static_cast<int64_t>(s.mantissa) * static_cast<int64_t>(c_inv.mantissa);
		rm = static_cast<int32_t>(prod >> 29);
	}
	else {
		int32_t lo, hi;
		__asm__("smull %0, %1, %2, %3"
			: "=&r"(lo), "=&r"(hi)
			: "r"(s.mantissa), "r"(c_inv.mantissa));
		rm = (hi << 3) | (static_cast<uint32_t>(lo) >> 29);
	}
	uint32_t abs_m = abs32(rm);
	if (UNLIKELY(abs_m >= MANT_OVERFLOW)) {
		rm >>= 1;
		re += 1;
	}
	return from_raw_unchecked(rm, re);
}

constexpr SoftFloat SoftFloat::sin() const noexcept {
	return sincos().sin;
}

constexpr SoftFloat SoftFloat::cos() const noexcept {
	return sincos().cos;
}

[[nodiscard]] constexpr SF_HOT SF_FLATTEN
SoftFloat SoftFloat::asin() const noexcept
{
	SoftFloat x = *this;
	bool neg = x.is_negative();
	x = x.abs();
	if (UNLIKELY(x > SoftFloat::one())) return SoftFloat::zero();

	SoftFloat result = atan2(x, (SoftFloat::one() - x * x).sqrt());
	return neg ? -result : result;
}

[[nodiscard]] constexpr SF_HOT SF_FLATTEN
SoftFloat SoftFloat::acos() const noexcept
{
	SoftFloat ax = this->abs();
	if (UNLIKELY(ax > SoftFloat::one())) return SoftFloat::zero();

	SoftFloat y = (SoftFloat::one() - ax * ax).sqrt();
	SoftFloat r = atan2(y, ax);

	if (this->is_negative())
		r = SoftFloat::pi() - r;

	return r;
}

constexpr SoftFloat SoftFloat::sinh() const noexcept {
	SoftFloat e = exp();
	return (e - e.reciprocal()) >> 1;
}
constexpr SoftFloat SoftFloat::cosh() const noexcept {
	SoftFloat e = exp();
	return (e + e.reciprocal()) >> 1;
}
constexpr SoftFloat SoftFloat::tanh() const noexcept {
	SoftFloat e2 = (*this << 1).exp();
	SoftFloat num = e2 - SoftFloat::one();
	SoftFloat den = e2 + SoftFloat::one();
	return num * den.reciprocal();
}

constexpr SF_HOT SoftFloat SoftFloat::inv_sqrt() const noexcept
{
	if (UNLIKELY(mantissa <= 0)) return zero();

	const int32_t  E_raw = exponent + 29;
	const uint32_t a = static_cast<uint32_t>(mantissa);

	const uint32_t offset = a - MANT_MIN;
	const uint32_t idx = offset >> 21;
	const uint32_t frac8 = (offset >> 13) & 0xFFu;

	const int32_t v0 = INV_SQRT_Q29[idx];
	const int32_t v1 = INV_SQRT_Q29[idx + 1];
	int32_t y_q29 = v0 + (((v1 - v0) * static_cast<int32_t>(frac8)) >> 8);

	int32_t yy, ay, r_q29;

#if defined(__arm__)
	if (!SF_IS_CONSTEVAL()) {
		int32_t lo, hi;
		__asm__(
			"smull %0, %1, %2, %3"
			: "=&r"(lo),
			  "=&r"(hi)
			: "r"(y_q29),
			  "r"(y_q29));
		yy = (hi << 3) | (static_cast<uint32_t>(lo) >> 29);
	}
	else
#endif
		yy = static_cast<int32_t>(
			(static_cast<int64_t>(y_q29) * y_q29) >> 29);

#if defined(__arm__)
	if (!SF_IS_CONSTEVAL()) {
		int32_t lo, hi;
		__asm__(
			"smull %0, %1, %2, %3"
			: "=&r"(lo),
			  "=&r"(hi)
			: "r"(static_cast<int32_t>(a)),
			  "r"(yy));
		ay = (hi << 3) | (static_cast<uint32_t>(lo) >> 29);
	}
	else
#endif
		ay = static_cast<int32_t>(
			(static_cast<int64_t>(a) * yy) >> 29);

	r_q29 = 0x30000000 - (ay >> 1);

#if defined(__arm__)
	if (!SF_IS_CONSTEVAL()) {
		int32_t lo, hi;
		__asm__(
			"smull %0, %1, %2, %3"
			: "=&r"(lo),
			  "=&r"(hi)
			: "r"(y_q29),
			  "r"(r_q29));
		y_q29 = (hi << 3) | (static_cast<uint32_t>(lo) >> 29);
	}
	else
#endif
		y_q29 = static_cast<int32_t>(
			(static_cast<int64_t>(y_q29) * r_q29) >> 29);

	if (E_raw & 1) {
#if defined(__arm__)
		if (!SF_IS_CONSTEVAL()) {
			int32_t lo, hi;
			__asm__(
				"smull %0, %1, %2, %3"
				: "=&r"(lo), "=&r"(hi)
				: "r"(y_q29), "r"(0x16A09E66));
			y_q29 = (hi << 3) | (static_cast<uint32_t>(lo) >> 29);
		}
		else
#endif
			y_q29 = static_cast<int32_t>( (static_cast<int64_t>(y_q29) * 0x16A09E66LL) >> 29);
	}

	const int32_t carry = static_cast<uint32_t>(y_q29) >> 29;
	y_q29 <<= 1;
	int32_t result_e = -29 - (E_raw >> 1) - 1 + carry;

	result_e = sat_exp(result_e);

	return from_raw_unchecked(y_q29, result_e);
}

constexpr SF_HOT SoftFloat SoftFloat::sqrt() const noexcept
{
	if (UNLIKELY(mantissa <= 0)) return zero();

	if (SF_IS_CONSTEVAL()) {
		int32_t m = mantissa;
		int32_t e = exponent;
		if (e & 1) { m <<= 1; e -= 1; }
		uint64_t scaled = static_cast<uint64_t>(m) << 30;
		uint64_t root = isqrt64(scaled);
		int32_t  rm = static_cast<int32_t>(root);
		int32_t  re = e / 2 - 15;
		return SoftFloat(rm, re);
	}

	const int32_t  E_raw = exponent + 29;
	const uint32_t a = static_cast<uint32_t>(mantissa);

	const uint32_t offset = a - MANT_MIN;
	const uint32_t idx = offset >> 21;
	const uint32_t frac8 = (offset >> 13) & 0xFFu;

	const int32_t v0 = INV_SQRT_Q29[idx];
	const int32_t v1 = INV_SQRT_Q29[idx + 1];
	int32_t y_q29 = v0 + (((v1 - v0) * static_cast<int32_t>(frac8)) >> 8);

	int32_t g_q29;
#if defined(__arm__)
	{
		int32_t lo, hi;
		__asm__(
			"smull %0, %1, %2, %3"
			: "=&r"(lo), "=&r"(hi)
			: "r"(static_cast<int32_t>(a)), "r"(y_q29));
		g_q29 = (hi << 3) | (static_cast<uint32_t>(lo) >> 29);
	}
#else
	g_q29 = static_cast<int32_t>(
		(static_cast<uint64_t>(a) * static_cast<uint32_t>(y_q29)) >> 29);
#endif

	int32_t r_q30;
#if defined(__arm__)
	{
		int32_t lo, hi;
		__asm__(
			"smull %0, %1, %2, %3"
			: "=&r"(lo), "=&r"(hi)
			: "r"(y_q29), "r"(g_q29));
		r_q30 = static_cast<int32_t>(0x20000000u)
			- ((hi << 3) | (static_cast<uint32_t>(lo) >> 29));
	}
#else
	{
		const int64_t yg = static_cast<int64_t>(y_q29) * g_q29;
		r_q30 = static_cast<int32_t>(0x20000000u) - static_cast<int32_t>(yg >> 29);
	}
#endif

#if defined(__arm__)
	{
		int32_t lo, hi;
		__asm__(
			"smull %0, %1, %2, %3"
			: "=&r"(lo), "=&r"(hi)
			: "r"(g_q29), "r"(r_q30));
		g_q29 += (hi << 2) | (static_cast<uint32_t>(lo) >> 30);
	}
#else
	{
		const int64_t gr = static_cast<int64_t>(g_q29) * r_q30;
		g_q29 += static_cast<int32_t>(gr >> 30);
	}
#endif

	if (E_raw & 1) {
#if defined(__arm__)
		int32_t lo, hi;
		__asm__(
			"smull %0, %1, %2, %3"
			: "=&r"(lo), "=&r"(hi)
			: "r"(g_q29), "r"(0x2D413CCD));
		g_q29 = (hi << 3) | (static_cast<uint32_t>(lo) >> 29);
#else
		g_q29 = static_cast<int32_t>(
			(static_cast<int64_t>(g_q29) * 0x2D413CCDLL) >> 29);
#endif
	}

	const int32_t result_e = sat_exp((E_raw >> 1) - 29);
	return from_raw_unchecked(g_q29, result_e);
}

constexpr SF_HOT SoftFloat atan2(SoftFloat y, SoftFloat x) noexcept
{
	static constexpr int32_t ATAN_TAB_Q29[257] = {
		       0,    2097141,    4194219,    6291168,
		 8387925,   10484427,   12580609,   14676407,
		16771758,   18866598,   20960863,   23054490,
		25147416,   27239578,   29330911,   31421354,
		33510843,   35599317,   37686712,   39772966,
		41858018,   43941805,   46024266,   48105340,
		50184965,   52263081,   54339626,   56414542,
		58487768,   60559244,   62628910,   64696708,
		66762579,   68826465,   70888307,   72948048,
		75005631,   77060998,   79114093,   81164859,
		83213242,   85259186,   87302634,   89343535,
		91381832,   93417472,   95450402,   97480570,
		99507923,  101532409,  103553977,  105572575,
	   107588154,  109600664,  111610055,  113616278,
	   115619285,  117619027,  119615459,  121608532,
	   123598200,  125584418,  127567140,  129546321,
	   131521918,  133493887,  135462185,  137426768,
	   139387596,  141344627,  143297819,  145247133,
	   147192530,  149133969,  151071412,  153004822,
	   154934160,  156859391,  158780477,  160697384,
	   162610076,  164518518,  166422677,  168322519,
	   170218011,  172109122,  173995820,  175878074,
	   177755853,  179629127,  181497868,  183362046,
	   185221634,  187076603,  188926928,  190772581,
	   192613537,  194449771,  196281257,  198107973,
	   199929894,  201746997,  203559260,  205366662,
	   207169181,  208966795,  210759486,  212547234,
	   214330019,  216107822,  217880627,  219648415,
	   221411170,  223168875,  224921514,  226669072,
	   228411535,  230148887,  231881116,  233608207,
	   235330149,  237046928,  238758533,  240464953,
	   242166178,  243862195,  245552997,  247238573,
	   248918915,  250594014,  252263862,  253928451,
	   255587776,  257241828,  258890602,  260534092,
	   262172294,  263805201,  265432810,  267055116,
	   268672116,  270283807,  271890185,  273491249,
	   275086997,  276677426,  278262536,  279842326,
	   281416795,  282985944,  284549771,  286108279,
	   287661468,  289209339,  290751894,  292289135,
	   293821065,  295347685,  296869000,  298385011,
	   299895724,  301401141,  302901268,  304396108,
	   305885667,  307369949,  308848960,  310322706,
	   311791193,  313254427,  314712414,  316165161,
	   317612676,  319054965,  320492037,  321923898,
	   323350557,  324772022,  326188302,  327599405,
	   329005341,  330406118,  331801746,  333192234,
	   334577593,  335957831,  337332960,  338702990,
	   340067931,  341427795,  342782591,  344132331,
	   345477027,  346816690,  348151331,  349480962,
	   350805596,  352125243,  353439918,  354749631,
	   356054396,  357354224,  358649130,  359939125,
	   361224223,  362504438,  363779782,  365050268,
	   366315911,  367576724,  368832721,  370083915,
	   371330321,  372571953,  373808825,  375040951,
	   376268345,  377491022,  378708997,  379922284,
	   381130898,  382334853,  383534165,  384728848,
	   385918917,  387104388,  388285275,  389461594,
	   390633360,  391800588,  392963293,  394121491,
	   395275197,  396424428,  397569197,  398709522,
	   399845417,  400976898,  402103981,  403226681,
	   404345015,  405458998,  406568646,  407673974,
	   408774999,  409871737,  410964203,  412052413,
	   413136383,  414216130,  415291668,  416363015,
	   417430186,  418493196,  419552063,  420606802,
	   421657428
	};

	if (UNLIKELY(x.mantissa == 0 && y.mantissa == 0))
		return SoftFloat::zero();

	const bool x_neg = x.mantissa < 0;
	const bool y_neg = y.mantissa < 0;

	if (UNLIKELY(y.mantissa == 0))
		return x_neg ? SoftFloat::pi() : SoftFloat::zero();
	if (UNLIKELY(x.mantissa == 0))
		return y_neg ? -SoftFloat::half_pi() : SoftFloat::half_pi();

	uint32_t ax = SoftFloat::abs32(x.mantissa);
	uint32_t ay = SoftFloat::abs32(y.mantissa);
	int32_t ex = x.exponent;
	int32_t ey = y.exponent;

	bool swap = false;
	uint32_t num = ay, den = ax;
	int32_t num_e = ey, den_e = ex;

	if (ey > ex || (ey == ex && ay > ax)) {
		swap = true;
		num = ax; den = ay;
		num_e = ex; den_e = ey;
	}

	int32_t shift = 24 + num_e - den_e;
	uint32_t t_Q24;

	if (LIKELY(shift >= 0)) {
		uint64_t n = static_cast<uint64_t>(num) << shift;
		t_Q24 = static_cast<uint32_t>(n / den);
	}
	else {
		uint32_t sn = num >> (-shift);
		t_Q24 = sn / den;
	}

	// Clamp: t_Q24 is 24-bit, table is 257 entries.
	if (UNLIKELY(t_Q24 >= 0x1000000u)) t_Q24 = 0xFFFFFFu;

	uint32_t idx = t_Q24 >> 16;
	uint32_t frac = t_Q24 & 0xFFFFu;

	int32_t a0 = ATAN_TAB_Q29[idx];
	int32_t a1 = ATAN_TAB_Q29[idx + 1];

	int64_t delta = static_cast<int64_t>(a1) - static_cast<int64_t>(a0);
	int32_t angle_q29;
#ifdef __arm__
	// On ARM, we can use SMULL to compute (delta * frac) >>
	if (SF_IS_CONSTEVAL()) {
		int64_t prod = delta * static_cast<int64_t>(frac);
		angle_q29 = a0 + static_cast<int32_t>(prod >> 16);
	}
	else {
		int32_t lo, hi;
		// Note: delta is int64_t, but we know it fits in 32 bits so we can use SMULL
		int32_t delta32 = static_cast<int32_t>(delta);
		__asm__("smull %0, %1, %2, %3"
			: "=&r"(lo), "=&r"(hi)
			: "r"(delta32), "r"(static_cast<int32_t>(frac)));
		angle_q29 = a0 + ((hi << 16) | (static_cast<uint32_t>(lo) >> 16));
	}
#else
	int64_t prod = delta * static_cast<int64_t>(frac);
	angle_q29 = a0 + static_cast<int32_t>(prod >> 16);
#endif

	constexpr int32_t HALF_PI_Q29 = 843314857;
	constexpr int32_t PI_Q29 = 1686629713;

	if (swap)  angle_q29 = HALF_PI_Q29 - angle_q29;
	if (x_neg) angle_q29 = PI_Q29 - angle_q29;
	if (y_neg) angle_q29 = -angle_q29;

	if (SF_IS_CONSTEVAL()) {
		return SoftFloat(angle_q29, -29);
	}

	int32_t m = angle_q29;
	int32_t e = -29;
	uint32_t a = m < 0 ? static_cast<uint32_t>(-m) : static_cast<uint32_t>(m);

	if (a >= SoftFloat::MANT_OVERFLOW) {
		m >>= 1;
		e += 1;
	}
	else if (a < SoftFloat::MANT_MIN && a != 0) {
		int lz = SoftFloat::clz(a);
		int s = lz - 2;
		m <<= s;
		e -= s;
	}
	return SoftFloat::from_raw_unchecked(m, e);
}

constexpr SF_HOT SoftFloat SoftFloat::exp() const noexcept
{
	static constexpr int32_t EXP_MANT[257] = {
		0x20000000, 0x201635F5, 0x202C7B54, 0x2042D028,
		0x2059347D, 0x206FA85C, 0x20862BD1, 0x209CBEE6,
		0x20B361A6, 0x20CA141C, 0x20E0D654, 0x20F7A857,
		0x210E8A31, 0x21257BED, 0x213C7D96, 0x21538F36,
		0x216AB0DA, 0x2181E28C, 0x21992457, 0x21B07646,
		0x21C7D866, 0x21DF4AC0, 0x21F6CD60, 0x220E6052,
		0x222603A0, 0x223DB757, 0x22557B81, 0x226D502A,
		0x2285355D, 0x229D2B27, 0x22B53191, 0x22CD48A9,
		0x22E57079, 0x22FDA90D, 0x2315F271, 0x232E4CB0,
		0x2346B7D7, 0x235F33F0, 0x2377C108, 0x23905F2A,
		0x23A90E63, 0x23C1CEBD, 0x23DAA046, 0x23F38308,
		0x240C7711, 0x24257C6B, 0x243E9323, 0x2457BB45,
		0x2470F4DD, 0x248A3FF7, 0x24A39C9F, 0x24BD0AE2,
		0x24D68ACC, 0x24F01C68, 0x2509BFC4, 0x252374EB,
		0x253D3BEA, 0x255714CE, 0x2570FFA2, 0x258AFC73,
		0x25A50B4E, 0x25BF2C3F, 0x25D95F52, 0x25F3A495,
		0x260DFC14, 0x262865DC, 0x2642E1F9, 0x265D7077,
		0x26781165, 0x2692C4CE, 0x26AD8ABF, 0x26C86346,
		0x26E34E6E, 0x26FE4C46, 0x27195CDA, 0x27348037,
		0x274FB66A, 0x276AFF80, 0x27865B86, 0x27A1CA8A,
		0x27BD4C98, 0x27D8E1BE, 0x27F48A09, 0x28104587,
		0x282C1444, 0x2847F64E, 0x2863EBB3, 0x287FF47F,
		0x289C10C1, 0x28B84085, 0x28D483DA, 0x28F0DACD,
		0x290D456C, 0x2929C3C3, 0x294655E2, 0x2962FBD5,
		0x297FB5AA, 0x299C8370, 0x29B96534, 0x29D65B04,
		0x29F364ED, 0x2A1082FF, 0x2A2DB546, 0x2A4AFBD0,
		0x2A6856AD, 0x2A85C5EA, 0x2AA34995, 0x2AC0E1BC,
		0x2ADE8E6D, 0x2AFC4FB8, 0x2B1A25A9, 0x2B381050,
		0x2B560FBB, 0x2B7423F7, 0x2B924D15, 0x2BB08B21,
		0x2BCEDE2B, 0x2BED4642, 0x2C0BC373, 0x2C2A55CE,
		0x2C48FD60, 0x2C67BA3A, 0x2C868C6A, 0x2CA573FD,
		0x2CC47105, 0x2CE3838E, 0x2D02ABA9, 0x2D21E963,
		0x2D413CCD, 0x2D60A5F5, 0x2D8024EA, 0x2D9FB9BC,
		0x2DBF6479, 0x2DDF2531, 0x2DFEFBF3, 0x2E1EE8CE,
		0x2E3EEBD2, 0x2E5F050E, 0x2E7F3491, 0x2E9F7A6C,
		0x2EBFD6AD, 0x2EE04963, 0x2F00D2A0, 0x2F217271,
		0x2F4228E8, 0x2F62F613, 0x2F83DA02, 0x2FA4D4C6,
		0x2FC5E66E, 0x2FE70F09, 0x30084EA8, 0x3029A55C,
		0x304B1333, 0x306C983D, 0x308E348C, 0x30AFE82F,
		0x30D1B337, 0x30F395B2, 0x31158FB3, 0x3137A149,
		0x3159CA84, 0x317C0B76, 0x319E642D, 0x31C0D4BC,
		0x31E35D32, 0x3205FDA0, 0x3228B617, 0x324B86A7,
		0x326E6F62, 0x32917057, 0x32B48998, 0x32D7BB35,
		0x32FB0540, 0x331E67C9, 0x3341E2E2, 0x3365769B,
		0x33892305, 0x33ACE833, 0x33D0C634, 0x33F4BD1A,
		0x3418CCF7, 0x343CF5DB, 0x346137D9, 0x34859301,
		0x34AA0764, 0x34CE9516, 0x34F33C26, 0x3517FCA8,
		0x353CD6AB, 0x3561CA42, 0x3586D780, 0x35ABFE74,
		0x35D13F33, 0x35F699CC, 0x361C0E53, 0x36419CD9,
		0x36674571, 0x368D082B, 0x36B2E51C, 0x36D8DC54,
		0x36FEEDE6, 0x372519E4, 0x374B6061, 0x3771C16F,
		0x37983D21, 0x37BED388, 0x37E584B8, 0x380C50C3,
		0x383337BB, 0x385A39B4, 0x388156C0, 0x38A88EF2,
		0x38CFE25D, 0x38F75113, 0x391EDB28, 0x394680AF,
		0x396E41BA, 0x39961E5D, 0x39BE16AB, 0x39E62AB7,
		0x3A0E5A94, 0x3A36A656, 0x3A5F0E10, 0x3A8791D6,
		0x3AB031BA, 0x3AD8EDD1, 0x3B01C62E, 0x3B2ABAE4,
		0x3B53CC08, 0x3B7CF9AC, 0x3BA643E6, 0x3BCFAAC8,
		0x3BF92E67, 0x3C22CED6, 0x3C4C8C2A, 0x3C766676,
		0x3CA05DCF, 0x3CCA7249, 0x3CF4A3F8, 0x3D1EF2F0,
		0x3D495F45, 0x3D73E90D, 0x3D9E905B, 0x3DC95544,
		0x3DF437DD, 0x3E1F3839, 0x3E4A566F, 0x3E759292,
		0x3EA0ECB7, 0x3ECC64F3, 0x3EF7FB5B, 0x3F23B004,
		0x3F4F8303, 0x3F7B746D, 0x3FA78457, 0x3FD3B2D6,
		0x20000000
	};

	if (UNLIKELY(mantissa == 0)) return one();

	constexpr int32_t INV_LN2_M = 0x2E2A8ECB;

	int64_t kprod;
#if defined(__arm__)
	if (!SF_IS_CONSTEVAL()) {
		int32_t lo, hi;
		__asm__(
			"smull %0, %1, %2, %3"
			: "=&r"(lo), "=&r"(hi)
			: "r"(mantissa), "r"(INV_LN2_M));
		kprod = (static_cast<int64_t>(hi) << 32) | static_cast<uint32_t>(lo);
	}
	else
#endif
	{
		kprod = static_cast<int64_t>(mantissa) * static_cast<int64_t>(INV_LN2_M);
	}

	const int32_t k_rshift = 29 - exponent;
	if (UNLIKELY(k_rshift <= 0))
		return mantissa > 0 ? from_raw_unchecked(MANT_MIN, EXP_MAX) : zero();

	int32_t  k;
	uint32_t u_8_21;

	if (LIKELY(k_rshift <= 63)) {
		k = static_cast<int32_t>(kprod >> k_rshift);

		const uint64_t mask = (uint64_t(1) << k_rshift) - 1u;
		const uint64_t frac_bits = static_cast<uint64_t>(kprod) & mask;

		u_8_21 = (k_rshift > 29)
			? static_cast<uint32_t>(frac_bits >> (k_rshift - 29))
			: static_cast<uint32_t>(frac_bits) << (29 - k_rshift);
	}
	else {
		k = (kprod < 0) ? -1 : 0;
		const uint32_t rsh = static_cast<uint32_t>(k_rshift - 29);
		u_8_21 = (rsh < 60)
			? static_cast<uint32_t>(static_cast<uint64_t>(kprod) >> rsh)
			: 0u;
	}

	const uint32_t idx = u_8_21 >> 21;
	const int32_t  frac = static_cast<int32_t>(u_8_21 & 0x1FFFFFu);

	const int32_t m0 = EXP_MANT[idx];
	const int32_t m1 = LIKELY(idx < 255) ? EXP_MANT[idx + 1] : int32_t(0x40000000);
	const int32_t delta = m1 - m0;

	int32_t result_q29;
#if defined(__arm__)
	if (!SF_IS_CONSTEVAL()) {
		int32_t lo, hi;
		__asm__(
			"smull %0, %1, %2, %3"
			: "=&r"(lo), "=&r"(hi)
			: "r"(delta), "r"(frac));
		result_q29 = m0 + ((hi << 11) | (static_cast<uint32_t>(lo) >> 21));
	}
	else
#endif
	{
		result_q29 = m0 + static_cast<int32_t>((static_cast<int64_t>(delta) * frac) >> 21);
	}

	const int32_t final_exp = k - 29;
	if (UNLIKELY(final_exp > EXP_MAX))  return from_raw_unchecked(MANT_MIN, EXP_MAX);
	if (UNLIKELY(final_exp < EXP_MIN)) return zero();
	return from_raw_unchecked(result_q29, final_exp);
}

constexpr SF_HOT SoftFloat SoftFloat::log2() const noexcept
{
	static constexpr int32_t LOG2_Q30[257] = {
		0x00000000, 0x005C2711, 0x00B7F285, 0x01136311,
		0x016E7968, 0x01C9363B, 0x02239A3A, 0x027DA612,
		0x02D75A6E, 0x0330B7F8, 0x0389BF57, 0x03E27130,
		0x043ACE27, 0x0492D6DF, 0x04EA8BF7, 0x0541EE0D,
		0x0598FDBE, 0x05EFBBA5, 0x0646285B, 0x069C4477,
		0x06F21090, 0x07478D38, 0x079CBB04, 0x07F19A83,
		0x08462C46, 0x089A70DA, 0x08EE68CB, 0x094214A5,
		0x099574F1, 0x09E88A36, 0x0A3B54FC, 0x0A8DD5C8,
		0x0AE00D1C, 0x0B31FB7D, 0x0B83A16A, 0x0BD4FF63,
		0x0C2615E8, 0x0C76E574, 0x0CC76E83, 0x0D17B191,
		0x0D67AF16, 0x0DB7678B, 0x0E06DB66, 0x0E560B1E,
		0x0EA4F726, 0x0EF39FF1, 0x0F4205F3, 0x0F90299C,
		0x0FDE0B5C, 0x102BABA2, 0x107908DB, 0x10C62975,
		0x111307DA, 0x115FA676, 0x11AC05B2, 0x11F825F6,
		0x124407AB, 0x128FAB35, 0x12DB10FC, 0x13263963,
		0x13712ACE, 0x13BBD3A0, 0x1406463B, 0x14507CFE,
		0x149A784B, 0x14E43880, 0x152DBDFC, 0x1577091B,
		0x15C01A39, 0x1608F1B4, 0x16518FE4, 0x1699F524,
		0x16E221CD, 0x172A1637, 0x1771D2BA, 0x17B957AC,
		0x1800A563, 0x1847BC33, 0x188E9C72, 0x18D54673,
		0x191BBA89, 0x1961F905, 0x19A80239, 0x19EDD675,
		0x1A33760A, 0x1A78E146, 0x1ABE1879, 0x1B031BEF,
		0x1B47EBF7, 0x1B8C88DB, 0x1BD0F2E9, 0x1C152A6C,
		0x1C592FAD, 0x1C9D02F6, 0x1CE0A492, 0x1D2414C8,
		0x1D6753E0, 0x1DAA6222, 0x1DED3FD4, 0x1E2FED3D,
		0x1E726AA1, 0x1EB4B847, 0x1EF6D673, 0x1F38C567,
		0x1F7A8568, 0x1FBC16B9, 0x1FFD799A, 0x203EAE4E,
		0x207FB517, 0x20C08E33, 0x210139E4, 0x2141B869,
		0x21820A01, 0x21C22EEA, 0x22022762, 0x2241F3A7,
		0x228193F5, 0x22C10889, 0x2300519E, 0x233F6F71,
		0x237E623D, 0x23BD2A3B, 0x23FBC7A6, 0x243A3AB7,
		0x247883A8, 0x24B6A2B1, 0x24F4980B, 0x253263EC,
		0x2570068E, 0x25AD8026, 0x25EAD0EB, 0x2627F914,
		0x2664F8D5, 0x26A1D064, 0x26DE7FF6, 0x271B07C0,
		0x275767F5, 0x2793A0C9, 0x27CFB26F, 0x280B9D1A,
		0x284760FD, 0x2882FE49, 0x28BE7531, 0x28F9C5E5,
		0x2934F097, 0x296FF577, 0x29AAD4B6, 0x29E58E83,
		0x2A20230E, 0x2A5A9285, 0x2A94DD19, 0x2ACF02F7,
		0x2B09044D, 0x2B42E149, 0x2B7C9A19, 0x2BB62EEA,
		0x2BEF9FE8, 0x2C28ED40, 0x2C62171E, 0x2C9B1DAE,
		0x2CD4011C, 0x2D0CC192, 0x2D455F3C, 0x2D7DDA44,
		0x2DB632D4, 0x2DEE6917, 0x2E267D36, 0x2E5E6F5A,
		0x2E963FAC, 0x2ECDEE56, 0x2F057B7F, 0x2F3CE751,
		0x2F7431F2, 0x2FAB5B8B, 0x2FE26443, 0x30194C40,
		0x305013AB, 0x3086BAA9, 0x30BD4161, 0x30F3A7F8,
		0x3129EE96, 0x3160155E, 0x31961C76, 0x31CC0404,
		0x3201CC2C, 0x32377512, 0x326CFEDB, 0x32A269AB,
		0x32D7B5A5, 0x330CE2ED, 0x3341F1A7, 0x3376E1F5,
		0x33ABB3FA, 0x33E067D9, 0x3414FDB4, 0x344975AD,
		0x347DCFE7, 0x34B20C82, 0x34E62BA0, 0x351A2D62,
		0x354E11EB, 0x3581D959, 0x35B583CE, 0x35E9116A,
		0x361C824D, 0x364FD697, 0x36830E69, 0x36B629E1,
		0x36E9291E, 0x371C0C41, 0x374ED367, 0x37817EAF,
		0x37B40E39, 0x37E68222, 0x3818DA88, 0x384B178A,
		0x387D3945, 0x38AF3FD7, 0x38E12B5D, 0x3912FBF4,
		0x3944B1B9, 0x39764CC9, 0x39A7CD41, 0x39D9333D,
		0x3A0A7EDA, 0x3A3BB033, 0x3A6CC764, 0x3A9DC48A,
		0x3ACEA7C0, 0x3AFF7121, 0x3B3020C8, 0x3B60B6D1,
		0x3B913356, 0x3BC19672, 0x3BF1E041, 0x3C2210DB,
		0x3C52285C, 0x3C8226DD, 0x3CB20C79, 0x3CE1D948,
		0x3D118D66, 0x3D4128EB, 0x3D70ABF1, 0x3DA01691,
		0x3DCF68E3, 0x3DFEA301, 0x3E2DC503, 0x3E5CCF02,
		0x3E8BC117, 0x3EBA9B59, 0x3EE95DE1, 0x3F1808C7,
		0x3F469C22, 0x3F75180B, 0x3FA37C98, 0x3FD1C9E2,
		0x40000000
	};

	if (UNLIKELY(mantissa <= 0)) return zero();

	int32_t  E = exponent + 29;
	uint32_t m_abs = static_cast<uint32_t>(mantissa);

	uint32_t low = m_abs - MANT_MIN;
	uint32_t t_int = low >> 21;
	uint32_t frac = (low >> 13) & 0xFFu;

	int32_t v0 = LOG2_Q30[t_int];
	int32_t v1 = LOG2_Q30[t_int + 1];

	int32_t delta = v1 - v0;
	int32_t corr = (delta * static_cast<int32_t>(frac)) >> 8;
	int32_t log2_frac_q30 = v0 + corr;

	SoftFloat fractional_part(log2_frac_q30, -30);
	SoftFloat integer_part(E);
	return fractional_part + integer_part;
}

constexpr SoftFloat SoftFloat::log() const noexcept {
	constexpr SoftFloat LN2 = from_raw_unchecked(0x2C5C85FE, -30);
	return log2() * LN2;
}

constexpr SoftFloat SoftFloat::log10() const noexcept {
	constexpr SoftFloat LOG10_2 = SoftFloat::from_raw_unchecked(0x268826A1, -31);
	return log2() * LOG10_2;
}

constexpr SoftFloat SoftFloat::pow(SoftFloat y) const noexcept {
	if (mantissa == 0) return y.mantissa == 0 ? one() : zero();
	if (y.mantissa == 0) return one();

	if (y == one())       return *this;
	if (y == two())       { SoftFloat t = *this; return SoftFloat::mul_plain(t, t); }
	if (y == three())     { SoftFloat t = *this; return SoftFloat::mul_plain(t, SoftFloat::mul_plain(t, t)); }
	if (y == four())      { SoftFloat t = *this; t = SoftFloat::mul_plain(t, t); return SoftFloat::mul_plain(t, t); }
	if (y == neg_one())   return reciprocal();
	if (y == half())      return sqrt();
	if (y == -half())     return inv_sqrt();

	if (y.mantissa == 0x30000000 && y.exponent == -29) {
		SoftFloat t = sqrt();
		return SoftFloat::mul_plain(*this, t);
	}
	if (y.mantissa == 0x20000000 && y.exponent == -31)
		return sqrt().sqrt();

	int32_t n = 0;
	bool is_int = false;
	if (y.exponent < 0) {
		int32_t shift = -y.exponent;
		if (shift <= 30) {
			uint32_t a = abs32(y.mantissa);
			uint32_t mask = (1u << shift) - 1u;
			if ((a & mask) == 0u) {
				is_int = true;
				n = static_cast<int32_t>(a >> shift);
				if (y.mantissa < 0) n = -n;
			}
		}
	}

	if (is_int) {
		if (n == 0) return one();
		if (n == 1) return *this;
		if (n == -1) return reciprocal();

		bool neg = n < 0;
		uint32_t un = neg
		    ? static_cast<uint32_t>(-(static_cast<int64_t>(n)))
		    : static_cast<uint32_t>(n);

		if (exponent > 0 && un > 127u / static_cast<uint32_t>(exponent))
			return from_raw_unchecked(mantissa > 0 ? MANT_MIN : -static_cast<int32_t>(MANT_MIN), EXP_MAX);
		if (exponent < 0 && un > 128u / static_cast<uint32_t>(-exponent))
			return zero();

		SoftFloat result = one();
		SoftFloat base   = *this;
		for (; un; un >>= 1) {
			if (un & 1u) result = SoftFloat::mul_plain(result, base);
			if (un == 1u) break;
			base = SoftFloat::mul_plain(base, base);
		}
		return neg ? result.reciprocal() : result;
	}

	if (is_negative()) return zero();
	return (y * log()).exp();
}

constexpr SF_HOT SoftFloat hypot(SoftFloat x, SoftFloat y) noexcept {
	x = x.abs();
	y = y.abs();

	if (x.mantissa == 0) return y;
	if (y.mantissa == 0) return x;
	if (x < y) { SoftFloat t = x; x = y; y = t; }

	int32_t ex = x.exponent;
	int32_t d  = ex - y.exponent;

	if (d >= 15) return x;

	uint32_t mx = static_cast<uint32_t>(x.mantissa);
	uint32_t my = static_cast<uint32_t>(y.mantissa);

	uint64_t mx2 = static_cast<uint64_t>(mx) * mx;
	uint64_t my2 = static_cast<uint64_t>(my) * my;
	uint64_t S   = mx2 + (my2 >> (2 * d));

	uint32_t s_hi = static_cast<uint32_t>(S >> 29);
	int32_t  s_e  = 29;

	if (s_hi >= 0x80000000u) {
		s_hi >>= 2;
		s_e  += 2;
	}
	else if (s_hi >= SoftFloat::MANT_OVERFLOW) {
		s_hi >>= 1;
		s_e  += 1;
	}

	SoftFloat s_sf = SoftFloat::from_raw_unchecked(static_cast<int32_t>(s_hi), s_e);

	SoftFloat r = s_sf.sqrt();
	return SoftFloat::from_raw_unchecked(r.mantissa, r.exponent + ex);
}

constexpr SF_HOT SoftFloat SoftFloat::trunc() const noexcept {
	return SoftFloat(to_int32());
}

constexpr SF_HOT SoftFloat SoftFloat::floor() const noexcept {
	if (UNLIKELY(mantissa == 0)) return *this;
	if (exponent >= 0) return *this;

	int32_t rs = -exponent;

	if (rs >= 31) {
		return mantissa > 0 ? SoftFloat::zero() : SoftFloat::neg_one();
	}

	uint32_t a = abs32(mantissa);

	uint32_t frac_mask = (1u << rs) - 1u;
	bool has_frac = (a & frac_mask) != 0;

	int32_t int_part_m = static_cast<int32_t>(a >> rs);
	int32_t result_m;

	if (mantissa < 0) {
		if (has_frac) {
			result_m = -(int_part_m + 1);
		} else {
			result_m = -int_part_m;
		}
	} else {
		result_m = int_part_m;
	}

	return SoftFloat(result_m);
}

constexpr SoftFloat SoftFloat::ceil() const noexcept {
	if (UNLIKELY(mantissa == 0)) return *this;
	if (exponent >= 0) return *this;

	int32_t rs = -exponent;
	if (rs >= 30) {                 // all fractional
		return is_positive() ? one() : zero();
	}

	uint32_t a = abs32(mantissa);
	uint32_t frac_mask = (1u << rs) - 1u;
	bool has_frac = (a & frac_mask) != 0;
	uint32_t int_part = a & ~frac_mask;

	if (mantissa < 0) {
		return SoftFloat(-static_cast<int32_t>(int_part), exponent);
	}
	else {
		uint32_t new_m = int_part;
		if (has_frac) new_m += (1u << rs);   // increase integer part by 1
		if (new_m >= MANT_OVERFLOW) {
			// rare, but handle overflow: e.g., int_part == 0x3FFFFFFF and has_frac
			// then new_m = 0x40000000 → unnormalised; the SoftFloat constructor will normalise.
			return SoftFloat(static_cast<int32_t>(new_m), exponent);
		}
		return SoftFloat(static_cast<int32_t>(new_m), exponent);
	}
}

constexpr SF_HOT SoftFloat SoftFloat::round() const noexcept {
	if (UNLIKELY(mantissa == 0)) return *this;
	if (exponent >= 0) return *this;

	int32_t rs = -exponent;

	if (rs >= 31) return SoftFloat::zero();

	uint32_t a = abs32(mantissa);

	uint32_t bias = 1u << (rs - 1);
	uint32_t sum = a + bias;

	int32_t result_m = static_cast<int32_t>(sum >> rs);

	if (mantissa < 0) result_m = -result_m;

	return SoftFloat(result_m);
}

constexpr SoftFloat SoftFloat::fract() const noexcept {
	return *this - trunc();
}

constexpr IntFractPair SoftFloat::modf() const noexcept {
	SoftFloat intpart = trunc();
	return { intpart, *this - intpart };
}

constexpr SoftFloat SoftFloat::copysign(SoftFloat sign) const noexcept {
	return from_raw_unchecked((mantissa ^ sign.mantissa) >= 0 ? mantissa : -mantissa, exponent);
}

constexpr SoftFloat SoftFloat::fmod(SoftFloat y) const noexcept {
	if (UNLIKELY(y.mantissa == 0)) return *this;
	if (UNLIKELY(mantissa == 0))   return *this;

	int32_t  sx = (mantissa < 0) ? -1 : 1;
	uint32_t ax = abs32(mantissa);
	uint32_t ay = abs32(y.mantissa);
	int32_t  d = exponent - y.exponent;

	if (d < 0) return *this;
	if (d == 0 && ax < ay) return *this;

	// Fast path for d == 0
	if (d == 0) {
		uint32_t r = ax - ay;
		if (r == 0) return zero();
		int32_t rm = static_cast<int32_t>(r);
		int32_t re = y.exponent;
		normalise_fast(rm, re);
		return from_raw_unchecked(sx * rm, re);
	}

	// Compute r = (ax * 2^d) % ay using binary exponentiation of 2^d mod ay
	uint32_t ax_mod = ax % ay;
	if (ax_mod == 0) return zero();

	// Compute 2^d mod ay
	uint32_t pow2_mod = 1;
	uint32_t base = 2 % ay;
	uint32_t remaining = static_cast<uint32_t>(d);
	while (remaining) {
		if (remaining & 1) {
			// pow2_mod = (pow2_mod * base) % ay
			uint64_t prod = static_cast<uint64_t>(pow2_mod) * base;
			uint32_t q, rem;
			if !consteval {
				__asm__("umull %0, %1, %2, %3"
					: "=&r"(rem), "=&r"(q)   // lo = rem, hi = q
					: "r"(pow2_mod), "r"(base));
				// remainder = (q*2^32 + rem) % ay → need two UDIVs
				uint32_t r_hi = q / ay;
				uint32_t r_lo_part = q - r_hi * ay;
				uint64_t combined = (static_cast<uint64_t>(r_lo_part) << 32) | rem;
				pow2_mod = static_cast<uint32_t>(combined % ay);
			}
			else {
				pow2_mod = static_cast<uint32_t>(prod % ay);
			}
		}
		// base = (base * base) % ay
		{
			uint64_t sq = static_cast<uint64_t>(base) * base;
			if !consteval {
				uint32_t sq_lo, sq_hi;
				__asm__("umull %0, %1, %2, %3"
					: "=&r"(sq_lo), "=&r"(sq_hi)
					: "r"(base), "r"(base));
				uint32_t r_hi = sq_hi / ay;
				uint32_t r_lo_part = sq_hi - r_hi * ay;
				uint64_t combined = (static_cast<uint64_t>(r_lo_part) << 32) | sq_lo;
				base = static_cast<uint32_t>(combined % ay);
			}
			else {
				base = static_cast<uint32_t>(sq % ay);
			}
		}
		remaining >>= 1;
	}

	// Final result = (ax_mod * pow2_mod) % ay
	uint64_t final_prod = static_cast<uint64_t>(ax_mod) * pow2_mod;
	uint32_t r;
	if !consteval {
		uint32_t lo, hi;
		__asm__("umull %0, %1, %2, %3"
			: "=&r"(lo), "=&r"(hi)
			: "r"(ax_mod), "r"(pow2_mod));
		uint32_t r_hi = hi / ay;
		uint32_t r_lo_part = hi - r_hi * ay;
		uint64_t combined = (static_cast<uint64_t>(r_lo_part) << 32) | lo;
		r = static_cast<uint32_t>(combined % ay);
	}
	else {
		r = static_cast<uint32_t>(final_prod % ay);
	}

	if (r == 0) return zero();
	int32_t rm = static_cast<int32_t>(r);
	int32_t re = y.exponent;
	normalise_fast(rm, re);
	return from_raw_unchecked(sx * rm, re);
}

constexpr SoftFloat SoftFloat::fma(SoftFloat b, SoftFloat c) const noexcept {
	return fused_mul_add(*this, b, c);
}

constexpr SF_HOT SoftFloat lerp(SoftFloat a, SoftFloat b, SoftFloat t) noexcept {
	return a + t * (b - a);
}

// =========================================================================
// Compile-time evaluation tests
// =========================================================================

[[nodiscard]] consteval bool ct_approx(float a, float b, int max_ulp = 16) {
	if (a == b) return true;
	uint32_t ua = SoftFloat::bitcast<uint32_t>(a);
	uint32_t ub = SoftFloat::bitcast<uint32_t>(b);
	if ((ua ^ ub) & 0x80000000u) return false;
	int32_t diff = static_cast<int32_t>(ua - ub);
	return (diff < 0 ? -diff : diff) <= max_ulp;
}

[[nodiscard]] consteval bool ct_is_normalized(int32_t mantissa, int32_t exponent) {
	if (mantissa == 0) return true;
	uint32_t a = SoftFloat::abs32(mantissa);
	int lz = SoftFloat::clz(a);
	return lz == 2 && (a & 0xC0000000u) == 0;
}

[[nodiscard]] consteval bool ct_float_eq(SoftFloat sf, float expected) {
	return SoftFloat::bitcast<uint32_t>(sf.to_float()) == SoftFloat::bitcast<uint32_t>(expected);
}

static_assert(SoftFloat::zero().is_zero(), "zero.is_zero");
static_assert(!SoftFloat::one().is_zero(), "one.not_zero");
static_assert(SoftFloat::one().to_float() == 1.0f, "one==1");
static_assert(SoftFloat::neg_one().to_float() == -1.0f, "neg_one==-1");
static_assert(SoftFloat::half().to_float() == 0.5f, "half==0.5");
static_assert(SoftFloat::two().to_float() == 2.0f, "two==2");
static_assert((SoftFloat::one() + SoftFloat::one()).to_float() == 2.0f, "1+1==2");
static_assert((SoftFloat::two() - SoftFloat::one()).to_float() == 1.0f, "2-1==1");
static_assert((SoftFloat::two()* SoftFloat::two()).to_float() == 4.0f, "2*2==4");
static_assert((SoftFloat::two() / SoftFloat::two()).to_float() == 1.0f, "2/2==1");
static_assert(SoftFloat::zero().sin().is_zero(), "sin(0)==0");
static_assert(SoftFloat::one() < SoftFloat::two(), "1<2");
static_assert(SoftFloat::neg_one() < SoftFloat::zero(), "-1<0");
static_assert(SoftFloat::one() == SoftFloat::one(), "1==1");

// Zero
static_assert(SoftFloat::zero().mantissa == 0);
static_assert(SoftFloat::zero().exponent == 0);
static_assert(SoftFloat::zero().is_zero());
static_assert(ct_float_eq(SoftFloat::zero(), 0.0f));

// One
static_assert(ct_is_normalized(SoftFloat::one().mantissa, SoftFloat::one().exponent));
static_assert(SoftFloat::one().mantissa == SoftFloat::MANT_MIN);
static_assert(SoftFloat::one().exponent == -SoftFloat::MANT_BITS);
static_assert(ct_float_eq(SoftFloat::one(), 1.0f));
static_assert(!SoftFloat::one().is_negative());

// Negative One
static_assert(ct_is_normalized(SoftFloat::neg_one().mantissa, SoftFloat::neg_one().exponent));
static_assert(SoftFloat::neg_one().mantissa == -static_cast<int32_t>(SoftFloat::MANT_MIN));
static_assert(SoftFloat::neg_one().exponent == -SoftFloat::MANT_BITS);
static_assert(ct_float_eq(SoftFloat::neg_one(), -1.0f));
static_assert(SoftFloat::neg_one().is_negative());

// Half (0.5)
static_assert(ct_is_normalized(SoftFloat::half().mantissa, SoftFloat::half().exponent));
static_assert(SoftFloat::half().mantissa == SoftFloat::MANT_MIN);
static_assert(SoftFloat::half().exponent == -SoftFloat::MANT_BITS - 1);
static_assert(ct_float_eq(SoftFloat::half(), 0.5f));

// Two (2.0)
static_assert(ct_is_normalized(SoftFloat::two().mantissa, SoftFloat::two().exponent));
static_assert(SoftFloat::two().mantissa == SoftFloat::MANT_MIN);
static_assert(SoftFloat::two().exponent == -SoftFloat::MANT_BITS + 1);
static_assert(ct_float_eq(SoftFloat::two(), 2.0f));

// Pi
static_assert(ct_is_normalized(SoftFloat::pi().mantissa, SoftFloat::pi().exponent));
static_assert(SoftFloat::pi().mantissa == 843314857);
static_assert(SoftFloat::pi().exponent == -28);
static_assert(ct_approx(SoftFloat::pi().to_float(), 3.14159265f, 2));
static_assert(SoftFloat::pi().is_positive());

// Two Pi
static_assert(ct_is_normalized(SoftFloat::two_pi().mantissa, SoftFloat::two_pi().exponent));
static_assert(SoftFloat::two_pi().mantissa == 843314857);
static_assert(SoftFloat::two_pi().exponent == -27);
static_assert(ct_approx(SoftFloat::two_pi().to_float(), 6.2831853f, 2));

// Half Pi
static_assert(ct_is_normalized(SoftFloat::half_pi().mantissa, SoftFloat::half_pi().exponent));
static_assert(SoftFloat::half_pi().mantissa == 843314857);
static_assert(SoftFloat::half_pi().exponent == -29);
static_assert(ct_approx(SoftFloat::two_pi().to_float(), 6.2831853f, 2));

// Relationships between constants
static_assert((SoftFloat::pi()* SoftFloat::two()).to_float() == SoftFloat::two_pi().to_float());
static_assert((SoftFloat::two_pi() / SoftFloat::two()).to_float() == SoftFloat::pi().to_float());
static_assert((SoftFloat::pi() / SoftFloat::two()).to_float() == SoftFloat::half_pi().to_float());
static_assert((SoftFloat::one() + SoftFloat::one()).to_float() == SoftFloat::two().to_float());
static_assert((SoftFloat::half() + SoftFloat::half()).to_float() == SoftFloat::one().to_float());
static_assert((-SoftFloat::one()).to_float() == SoftFloat::neg_one().to_float());

// Constants used in math functions
static_assert(ct_is_normalized(0x2C5C85FE, -30));
static_assert(ct_is_normalized(0x2E2B8A3E, -29));
static_assert(ct_is_normalized(0x2E2B8A3E, -21));
static_assert(ct_is_normalized(843314857, -36));
static_assert(ct_is_normalized(683565276, -23));
static_assert(ct_is_normalized(683565276, -32));

// Basic arithmetic
static_assert((SoftFloat::one() + SoftFloat::one()).to_float() == 2.0f);
static_assert((SoftFloat::two() - SoftFloat::one()).to_float() == 1.0f);
static_assert((SoftFloat::two()* SoftFloat::two()).to_float() == 4.0f);
static_assert((SoftFloat::two() / SoftFloat::two()).to_float() == 1.0f);
static_assert((-SoftFloat::one()).to_float() == -1.0f);
static_assert(SoftFloat::neg_one().abs().to_float() == 1.0f);

// Comparisons
static_assert(SoftFloat::one() < SoftFloat::two());
static_assert(SoftFloat::neg_one() < SoftFloat::zero());
static_assert(SoftFloat::one() == SoftFloat::one());
static_assert(SoftFloat::one() != SoftFloat::two());

// Shifts
static_assert((SoftFloat::one() << 2).to_float() == 4.0f);
static_assert((SoftFloat(8.0f) >> 2).to_float() == 2.0f);

// Fused operations
static_assert(fused_mul_add(SoftFloat::one(), SoftFloat::two(), SoftFloat::three()).to_float() == 7.0f);
static_assert(fused_mul_sub(SoftFloat::one(), SoftFloat::two(), SoftFloat::three()).to_float() == -5.0f);
static_assert(fused_mul_mul_add(SoftFloat::one(), SoftFloat::two(),
	SoftFloat::three(), SoftFloat::four()).to_float() == 14.0f);
static_assert(fused_mul_mul_sub(SoftFloat::one(), SoftFloat::two(),
	SoftFloat::three(), SoftFloat::four()).to_float() == -10.0f);

// Trigonometry
static_assert(SoftFloat::zero().sin().to_float() == 0.0f);
static_assert(ct_approx(SoftFloat::half_pi().sin().to_float(), 1.0f, 256));
static_assert(SoftFloat::zero().cos().to_float() == 1.0f);
static_assert(ct_approx(SoftFloat::pi().cos().to_float(), -1.0f, 1024));
static_assert(SoftFloat::zero().tan().to_float() == 0.0f);

static_assert(ct_approx(SoftFloat(0.0f).asin().to_float(), 0.0f, 2));
static_assert(ct_approx(SoftFloat(1.0f).asin().to_float(), 1.57079633f, 4));
static_assert(ct_approx(SoftFloat(-0.5f).asin().to_float(), -0.52359878f, 16));
static_assert(ct_approx(SoftFloat(0.5f).asin().to_float(), 0.52359878f, 16));
static_assert(ct_approx(SoftFloat(0.5f).acos().to_float(), 1.04719755f, 16));
static_assert(ct_approx(SoftFloat(-0.5f).acos().to_float(), 2.09439510f, 16));
static_assert(ct_approx(SoftFloat(0.0f).acos().to_float(), 1.57079633f, 4));
static_assert(ct_approx(SoftFloat(0.5f).acos().to_float(), 1.04719755f, 16));
static_assert(ct_approx(SoftFloat(1.0f).acos().to_float(), 0.0f, 2));
static_assert(ct_approx(SoftFloat(-0.5f).acos().to_float(), 2.09439510f, 4));
static_assert(ct_approx(SoftFloat(-1.0f).acos().to_float(), 3.14159265f, 4));

static_assert(atan(SoftFloat::zero()).to_float() == 0.0f);
static_assert(ct_approx(atan(SoftFloat::one()).to_float(), SoftFloat::half_pi().to_float() / 2.0f, 256));

// atan2
static_assert(atan2(SoftFloat::one(), SoftFloat::zero()).to_float() == SoftFloat::half_pi().to_float());
static_assert(ct_approx(atan2(SoftFloat::one(), SoftFloat::one()).to_float(), SoftFloat::half_pi().to_float() / 2.0f, 256));

// ---------- Exponential & Logarithm ----------
static_assert(SoftFloat::zero().exp().to_float() == 1.0f);
static_assert(ct_approx(SoftFloat::one().exp().to_float(), 2.7182818f, 512));
static_assert(SoftFloat::one().log().to_float() == 0.0f);
static_assert(ct_approx(SoftFloat::two().log().to_float(), 0.693147f, 512));
static_assert(SoftFloat::one().log2().to_float() == 0.0f);
static_assert(ct_approx(SoftFloat::two().log2().to_float(), 1.0f, 512));
static_assert(SoftFloat::one().log10().to_float() == 0.0f);
static_assert(ct_approx(SoftFloat(10.0f).log10().to_float(), 1.0f, 512));

// ---------- Power ----------
static_assert(SoftFloat::two().pow(SoftFloat::three()).to_float() == 8.0f);
static_assert(SoftFloat(4.0f).pow(SoftFloat::half()).to_float() == 2.0f);
static_assert(SoftFloat::zero().pow(SoftFloat::one()).to_float() == 0.0f);
static_assert(SoftFloat::one().pow(SoftFloat::zero()).to_float() == 1.0f);

// ---------- Square roots ----------
static_assert(SoftFloat(16.0f).sqrt().to_float() == 4.0f);
//static_assert(ct_approx(SoftFloat(2.0f).inv_sqrt().to_float(), 0.70710678f, 256));

// ---------- Rounding ----------
static_assert(SoftFloat(1.3f).trunc().to_float() == 1.0f);
static_assert(SoftFloat(-1.3f).trunc().to_float() == -1.0f);
static_assert(SoftFloat(1.3f).floor().to_float() == 1.0f);
static_assert(SoftFloat(-1.3f).floor().to_float() == -2.0f);
static_assert(SoftFloat(1.3f).ceil().to_float() == 2.0f);
static_assert(SoftFloat(-1.3f).ceil().to_float() == -1.0f);
static_assert(SoftFloat(1.5f).round().to_float() == 2.0f);
static_assert(SoftFloat(-1.5f).round().to_float() == -2.0f);
static_assert(SoftFloat(1.3f).fract().to_float() > 0.29f && SoftFloat(1.3f).fract().to_float() < 0.31f);

// ---------- modf ----------
static_assert([]() consteval {
	auto [i, f] = SoftFloat(1.3f).modf();
	return i.to_float() == 1.0f && f.to_float() > 0.29f && f.to_float() < 0.31f;
	}());

// ---------- Hyperbolic ----------
static_assert(SoftFloat::zero().sinh().to_float() == 0.0f);
static_assert(SoftFloat::zero().cosh().to_float() == 1.0f);
static_assert(SoftFloat::zero().tanh().to_float() == 0.0f);
static_assert(ct_approx(SoftFloat::one().sinh().to_float(), 1.175201f, 512));
static_assert(ct_approx(SoftFloat::one().cosh().to_float(), 1.543080f, 512));
static_assert(ct_approx(SoftFloat::one().tanh().to_float(), 0.761594f, 512));

// ---------- Sign manipulation ----------
static_assert(SoftFloat::one().copysign(SoftFloat::neg_one()).to_float() == -1.0f);
static_assert(SoftFloat::neg_one().copysign(SoftFloat::one()).to_float() == 1.0f);
static_assert(SoftFloat::neg_one().is_negative());

// ---------- Remainder ----------
static_assert(SoftFloat(5.3f).fmod(SoftFloat::two()).to_float() > 1.29f &&
	SoftFloat(5.3f).fmod(SoftFloat::two()).to_float() < 1.31f);
static_assert(SoftFloat(-5.3f).fmod(SoftFloat::two()).to_float() < -1.29f &&
	SoftFloat(-5.3f).fmod(SoftFloat::two()).to_float() > -1.31f);

// ---------- fma (member) ----------
static_assert(SoftFloat::two().fma(SoftFloat::three(), SoftFloat::four()).to_float() == 14.0f);

// ---------- Utility ----------
static_assert(min(SoftFloat::one(), SoftFloat::two()).to_float() == 1.0f);
static_assert(max(SoftFloat::one(), SoftFloat::two()).to_float() == 2.0f);
static_assert(clamp(SoftFloat::three(), SoftFloat::zero(), SoftFloat::two()).to_float() == 2.0f);
static_assert(lerp(SoftFloat::zero(), SoftFloat::two(), SoftFloat::half()).to_float() == 1.0f);
static_assert(hypot(SoftFloat::three(), SoftFloat::four()).to_float() == 5.0f);

// ---------- Expression template interactions ----------
static_assert((SoftFloat::one() + SoftFloat::two() * SoftFloat::three()).to_float() == 7.0f);
static_assert((SoftFloat::two() * SoftFloat::three() - SoftFloat::one()).to_float() == 5.0f);
static_assert((-(SoftFloat::two() * SoftFloat::three())).to_float() == -6.0f);

// FMA autodetection via += / -=
static_assert([]() consteval {
	SoftFloat a = SoftFloat::one();
	a += SoftFloat::two() * SoftFloat::three(); // 1 + 2*3 = 7
	return a.to_float() == 7.0f;
}());

static_assert([]() consteval {
	SoftFloat a = SoftFloat(10.0f);
	a -= SoftFloat::two() * SoftFloat::three(); // 10 - 2*3 = 4
	return a.to_float() == 4.0f;
}());

// Verify it's actually fused (same precision as explicit fused_mul_add)
static_assert([]() consteval {
	SoftFloat a = SoftFloat::one();
	SoftFloat b = SoftFloat::two();
	SoftFloat c = SoftFloat::three();
	SoftFloat r1 = a; r1 += b * c;
	SoftFloat r2 = fused_mul_add(a, b, c);
	return r1 == r2;
}());

// Missing constants
static_assert(SoftFloat::three().to_float() == 3.0f, "three == 3");
static_assert(SoftFloat::four().to_float() == 4.0f, "four == 4");

// Unary plus is identity
static_assert((+SoftFloat::one()).to_float() == 1.0f, "+one == 1");
static_assert((+SoftFloat::neg_one()).to_float() == -1.0f, "+neg_one == -1");

// Reciprocal
static_assert(ct_approx(SoftFloat::two().reciprocal().to_float(), 0.5f), "recip(2) ≈ 0.5");
static_assert((SoftFloat::half().reciprocal()).to_float() == 2.0f, "recip(0.5) == 2");
static_assert(reciprocal(SoftFloat::two()).to_float() == 0.5f, "free recip(2) == 0.5");

// atan (member)
static_assert(SoftFloat::zero().atan().to_float() == 0.0f, "atan(0) == 0");
static_assert(ct_approx(SoftFloat::one().atan().to_float(), SoftFloat::pi().to_float() / 4.0f, 256),
	"atan(1) ≈ pi/4");

// Mixed-type addition / subtraction / multiplication / division (with int/float)
static_assert((SoftFloat::one() + 2.0f).to_float() == 3.0f, "1 + 2.0f == 3");
static_assert((3.0f + SoftFloat::two()).to_float() == 5.0f, "3.0f + 2 == 5");
static_assert((SoftFloat::two() - 1).to_float() == 1.0f, "2 - 1 == 1");
static_assert((5 - SoftFloat::three()).to_float() == 2.0f, "5 - 3 == 2");
static_assert((SoftFloat::three() * 2.0f).to_float() == 6.0f, "3 * 2.0f == 6");
static_assert((4.0f * SoftFloat::one()).to_float() == 4.0f, "4.0f * 1 == 4");
static_assert((SoftFloat::one() * 10).to_float() == 10.0f, "1 * 10 == 10");
static_assert((10 * SoftFloat::one()).to_float() == 10.0f, "10 * 1 == 10");
static_assert(ct_approx((SoftFloat(12.0f) / 3).to_float(), 4.0f, 4), "12 / 3 ≈ 4");
static_assert(ct_approx((15.0f / SoftFloat::three()).to_float(), 5.0f, 4), "15.0f / 3 ≈ 5");

// Mixed comparisons
static_assert(SoftFloat::one() == 1.0f, "1 == 1.0f");
static_assert(1 == SoftFloat::one(), "1 == 1 (int)");
static_assert(SoftFloat::two() > 1.9f, "2 > 1.9");
static_assert(2 < SoftFloat::three(), "2 < 3");
static_assert(SoftFloat::neg_one() <= 0, "-1 <= 0");
static_assert(-1 <= SoftFloat::zero(), "-1 <= 0");
static_assert(SoftFloat::one() >= 0.5f, "1 >= 0.5");

// Assignment operators
static_assert([]() consteval {
	SoftFloat a;
	a = 42;          return a.to_float() == 42.0f;
	}(), "assign int");
static_assert([]() consteval {
	SoftFloat a;
	a = 3.125f;      return a.to_float() == 3.125f;
	}(), "assign float");
static_assert([]() consteval {
	SoftFloat a;
	a = int16_t(7);  return a.to_float() == 7.0f;
	}(), "assign int16_t");

// Compound assignment with MulExpr
static_assert([]() consteval {
	SoftFloat a(5.0f);
	a += SoftFloat::two() * SoftFloat::three();  // 5 + 6 = 11
	return a.to_float() == 11.0f;
	}(), "+= mul_expr");
static_assert([]() consteval {
	SoftFloat a(20.0f);
	a -= SoftFloat::two() * SoftFloat::three();  // 20 - 6 = 14
	return a.to_float() == 14.0f;
	}(), "-= mul_expr");

// MulExpr chaining: (a*b).sqrt(), etc.
static_assert(ct_approx(
	(SoftFloat::two()* SoftFloat::two()).sqrt().to_float(),
	2.0f, 4), "(2*2).sqrt() == 2");
static_assert(ct_approx(
	(SoftFloat::two()* SoftFloat::three()).exp().to_float(),
	403.42879349273512260838718054339f /*=expf(6.0f)*/, 1024), "(2*3).exp() ≈ exp(6)");
static_assert(ct_approx(
	(SoftFloat::two()* SoftFloat::four()).log2().to_float(),
	3.0f, 512), "(2*4).log2() == 3");

// User-defined literal
static_assert((1.5_sf).to_float() == 1.5f, "1.5_sf literal");
static_assert((3_sf).to_float() == 3.0f, "3_sf literal");
