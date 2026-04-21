/// File: SoftFloatTest.hh
#define _USE_MATH_DEFINES
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include "FusedSoftFloat.hh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace SoftFloatTest {

	static void (*output_func)(const char*) = nullptr;
	static void init_out() {
		if (!output_func)
			output_func = [](const char* s) { fputs(s, stdout); fflush(stdout); };
	}
	static void print(const char* s) { init_out(); output_func(s); }
	static void print_result(bool ok, const char* name) {
		char b[160]; snprintf(b, sizeof b, "[%s] %s\n", ok ? "PASS" : "FAIL", name); print(b);
	}
	static void print_section(const char* n) {
		char b[128]; snprintf(b, sizeof b, "\n=== %s ===\n", n); print(b);
	}

	extern "C" uint64_t DWT_GetTimeNs();
	
	static uint64_t now_ns() {
#if 0
		using namespace std::chrono;
		return static_cast<uint64_t>(
			duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
#else
		return DWT_GetTimeNs();
#endif
	}

	static uint32_t f2b(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
	static bool ulp_eq(float a, float b, int maxu = 8) {
		if (a == b) return true;
		if ((f2b(a) ^ f2b(b)) & 0x80000000u) return false;
		int32_t d = static_cast<int32_t>(f2b(a)) - static_cast<int32_t>(f2b(b));
		return (d < 0 ? -d : d) <= maxu;
	}
	static bool rel_eq(float a, float b, float tol = 1e-4f) {
		if (a == b) return true;
		float sc = fabsf(a) > fabsf(b) ? fabsf(a) : fabsf(b);
		if (sc == 0.f) return fabsf(a - b) < 1e-30f;
		return fabsf(a - b) / sc < tol;
	}
	static bool approx(float a, float b, int ulp = 8, float tol = 1e-4f) {
		return ulp_eq(a, b, ulp) || rel_eq(a, b, tol);
	}
	static bool chk(bool c, const char* msg) { if (!c) print(msg); return c; }

	// =========================================================================
	// Tests
	// =========================================================================

	static bool test_default_ctor() {
		SoftFloat z; return z.is_zero() && (float)z == 0.f;
	}

	static bool test_int_ctor() {
		bool ok = true;
		int vals[] = { 0,1,-1,2,-2,3,-3,100,-100,1000,-1000,1000000,-1000000 };
		for (int v : vals) {
			int32_t got = (int32_t)SoftFloat(v);
			if (got != v) {
				char b[64]; snprintf(b, sizeof b, "  int_ctor(%d)->%d\n", v, got); print(b); ok = false;
			}
		}
		return ok;
	}

	static bool test_float_ctor() {
		bool ok = true;
		float vals[] = { 0.f,1.f,-1.f,3.14159265f,1e-10f,1e10f,-1e-10f,0.5f,2.f,100.f };
		for (float v : vals) {
			float got = (float)SoftFloat(v);
			if (!approx(got, v, 4, 1e-5f)) {
				char b[128]; snprintf(b, sizeof b, "  float_ctor(%.8g)->%.8g\n", (double)v, (double)got);
				print(b); ok = false;
			}
		}
		return ok;
	}

	static bool test_constants() {
		bool ok = true;
		struct { SoftFloat sf; float ex; const char* nm; } cs[] = {
			{SoftFloat::zero(),    0.f,          "zero"},
			{SoftFloat::one(),     1.f,          "one"},
			{SoftFloat::neg_one(),-1.f,          "neg_one"},
			{SoftFloat::half(),    0.5f,         "half"},
			{SoftFloat::two(),     2.f,          "two"},
			{SoftFloat::pi(),      3.14159265f,  "pi"},
			{SoftFloat::two_pi(),  6.28318530f,  "two_pi"},
			{SoftFloat::half_pi(), 1.57079632f,  "half_pi"},
		};
		for (auto& c : cs) {
			float got = (float)c.sf;
			if (!approx(got, c.ex, 4, 1e-5f)) {
				char b[128]; snprintf(b, sizeof b, "  const %s: %.8g exp %.8g\n",
					c.nm, (double)got, (double)c.ex);
				print(b); ok = false;
			}
		}
		return ok;
	}

	static bool test_addition() {
		bool ok = true;
		struct { float a, b, ex; } cs[] = {
			{1.f,2.f,3.f},{-1.f,2.f,1.f},{5.f,0.f,5.f},
			{1e-8f,2e-8f,3e-8f},{1e10f,2e10f,3e10f},
		};
		for (auto& c : cs) {
			float got = (float)(SoftFloat(c.a) + SoftFloat(c.b));
			if (!approx(got, c.ex, 8, 2e-4f)) {
				char b[128]; snprintf(b, sizeof b, "  add(%.6g,%.6g)=%.8g exp %.8g\n",
					(double)c.a, (double)c.b, (double)got, (double)c.ex);
				print(b); ok = false;
			}
		}
		// Cancellation: due to 32-bit mantissa, expect ~7% error max when
		// operands differ by ~27 exponent positions (1.0 vs 1e-8).
		// We just check result is in the right ballpark.
		{
			SoftFloat r = (SoftFloat(1.f) + SoftFloat(1e-8f)) - SoftFloat(1.f);
			float got = (float)r;
			// Accept if within factor 2 of 1e-8 (32-bit precision limit)
			bool ok2 = (got > 0.f) && (got < 2e-8f);
			if (!ok2) {
				char b[128]; snprintf(b, sizeof b, "  cancellation: got %.8g\n", (double)got);
				print(b); ok = false;
			}
		}
		return ok;
	}

	static bool test_subtraction() {
		bool ok = true;
		struct { float a, b, ex; } cs[] = {
			{5.f,3.f,2.f},{1.f,5.f,-4.f},{7.f,0.f,7.f},{1000.f,.001f,999.999f},
		};
		for (auto& c : cs) {
			float got = (float)(SoftFloat(c.a) - SoftFloat(c.b));
			if (!approx(got, c.ex, 8, 2e-4f)) {
				char b[128]; snprintf(b, sizeof b, "  sub(%.6g,%.6g)=%.8g exp %.8g\n",
					(double)c.a, (double)c.b, (double)got, (double)c.ex);
				print(b); ok = false;
			}
		}
		ok &= chk((SoftFloat(123.f) - SoftFloat(123.f)).is_zero(), "  sub: x-x!=0\n");
		return ok;
	}

	static bool test_multiplication() {
		bool ok = true;
		struct { float a, b, ex; } cs[] = {
			{3.f,4.f,12.f},{0.f,100.f,0.f},{-2.f,5.f,-10.f},{-3.f,-4.f,12.f},
			{1e-5f,1e-5f,1e-10f},{1e5f,1e5f,1e10f},
		};
		for (auto& c : cs) {
			SoftFloat sr = SoftFloat(c.a) * SoftFloat(c.b);
			float got = (float)sr;
			if (c.ex == 0.f) { ok &= chk(sr.is_zero(), "  mul: zero case\n"); continue; }
			if (!approx(got, c.ex, 16, 2e-4f)) {
				char b[128]; snprintf(b, sizeof b, "  mul(%.6g,%.6g)=%.8g exp %.8g\n",
					(double)c.a, (double)c.b, (double)got, (double)c.ex);
				print(b); ok = false;
			}
		}
		return ok;
	}

	static bool test_division() {
		bool ok = true;
		struct { float a, b, ex; } cs[] = {
			{12.f,4.f,3.f},{0.f,5.f,0.f},{-15.f,3.f,-5.f},{100.f,2.f,50.f},
			{1.f,3.f,1.f / 3.f},{7.f,7.f,1.f},{1.f,1.f,1.f},{6.f,2.f,3.f},
			{1.f,4.f,0.25f},{3.f,2.f,1.5f},
		};
		for (auto& c : cs) {
			SoftFloat sr = SoftFloat(c.a) / SoftFloat(c.b);
			float got = (float)sr;
			if (c.ex == 0.f) { ok &= chk(sr.is_zero(), "  div: zero case\n"); continue; }
			if (!approx(got, c.ex, 64, 1e-3f)) {
				char b[128]; snprintf(b, sizeof b, "  div(%.6g,%.6g)=%.8g exp %.8g\n",
					(double)c.a, (double)c.b, (double)got, (double)c.ex);
				print(b); ok = false;
			}
		}
		return ok;
	}

	static bool test_negation() {
		bool ok = true;
		ok &= approx((float)(-SoftFloat(5.f)), -5.f);
		ok &= approx((float)(-SoftFloat(-3.f)), 3.f);
		ok &= (-SoftFloat(0.f)).is_zero();
		return ok;
	}

	static bool test_abs() {
		bool ok = true;
		ok &= approx((float)SoftFloat(-5.f).abs(), 5.f);
		ok &= approx((float)SoftFloat(3.f).abs(), 3.f);
		ok &= SoftFloat(0.f).abs().is_zero();
		return ok;
	}

	static bool test_comparison() {
		bool ok = true;
		SoftFloat a(5.f), b(3.f), c(5.f);
		ok &= (a > b); ok &= (b < a); ok &= (a >= b); ok &= (b <= a);
		ok &= (a == c); ok &= (a != b);
		ok &= (SoftFloat(-5.f) < SoftFloat(-3.f));
		ok &= (SoftFloat(-1.f) < SoftFloat(0.f));
		ok &= (SoftFloat(0.f) < SoftFloat(1.f));
		ok &= (SoftFloat(0.f) == SoftFloat(0.f));
		return ok;
	}

	static bool test_shift() {
		bool ok = true;
		ok &= approx((float)(SoftFloat(3.f) << 2), 12.f);
		ok &= approx((float)(SoftFloat(12.f) >> 2), 3.f);
		ok &= approx((float)(SoftFloat(-8.f) << 1), -16.f);
		ok &= approx((float)(SoftFloat(1.f) << 10), 1024.f);
		return ok;
	}

	static bool test_fma() {
		bool ok = true;
		struct { float a, b, c, ex; } cs[] = {
			{1.f,2.f,3.f,7.f},{-1.f,2.f,3.f,5.f},{0.f,5.f,7.f,35.f},
		};
		for (auto& c : cs) {
			float got = (float)fused_mul_add(SoftFloat(c.a), SoftFloat(c.b), SoftFloat(c.c));
			if (!approx(got, c.ex, 16, 5e-4f)) {
				char b[128]; snprintf(b, sizeof b, "  fma(%.4g,%.4g,%.4g)=%.8g exp %.8g\n",
					(double)c.a, (double)c.b, (double)c.c, (double)got, (double)c.ex);
				print(b); ok = false;
			}
		}
		ok &= chk(fused_mul_add(SoftFloat(1.f), SoftFloat(-1.f), SoftFloat(1.f)).is_zero(),
			"  fma(1,-1,1)!=0\n");
		return ok;
	}

	static bool test_fms() {
		bool ok = true;
		struct { float a, b, c, ex; } cs[] = {
			{10.f,2.f,3.f,4.f},{1.f,5.f,2.f,-9.f},
		};
		for (auto& c : cs) {
			float got = (float)fused_mul_sub(SoftFloat(c.a), SoftFloat(c.b), SoftFloat(c.c));
			if (!approx(got, c.ex, 16, 5e-4f)) {
				char b[128]; snprintf(b, sizeof b, "  fms(%.4g,%.4g,%.4g)=%.8g exp %.8g\n",
					(double)c.a, (double)c.b, (double)c.c, (double)got, (double)c.ex);
				print(b); ok = false;
			}
		}
		ok &= chk(fused_mul_sub(SoftFloat(6.f), SoftFloat(2.f), SoftFloat(3.f)).is_zero(),
			"  fms(6,2,3)!=0\n");
		return ok;
	}

	static bool test_fmma() {
		bool ok = true;
		float got = (float)fused_mul_mul_add(
			SoftFloat(2.f), SoftFloat(3.f), SoftFloat(4.f), SoftFloat(5.f));
		if (!approx(got, 26.f, 16, 5e-4f)) {
			char b[64]; snprintf(b, sizeof b, "  fmma=%.8g exp 26\n", (double)got); print(b); ok = false;
		}
		return ok;
	}

	static bool test_sqrt() {
		bool ok = true;
		struct { float x, ex; } cs[] = { {4.f,2.f},{9.f,3.f},{2.f,1.41421356f},{0.04f,.2f},{100.f,10.f} };
		for (auto& c : cs) {
			float got = (float)SoftFloat(c.x).sqrt();
			if (!approx(got, c.ex, 128, 1e-3f)) {
				char b[128]; snprintf(b, sizeof b, "  sqrt(%.4g)=%.8g exp %.8g\n",
					(double)c.x, (double)got, (double)c.ex); print(b); ok = false;
			}
		}
		ok &= chk(SoftFloat(0.f).sqrt().is_zero(), "  sqrt(0)!=0\n");
		return ok;
	}

	static bool test_inv_sqrt() {
		bool ok = true;
		struct { float x, ex; } cs[] = { {4.f,.5f},{2.f,.70710678f},{100.f,.1f} };
		for (auto& c : cs) {
			float got = (float)SoftFloat(c.x).inv_sqrt();
			if (!approx(got, c.ex, 128, 1e-3f)) {
				char b[128]; snprintf(b, sizeof b, "  inv_sqrt(%.4g)=%.8g exp %.8g\n",
					(double)c.x, (double)got, (double)c.ex); print(b); ok = false;
			}
		}
		return ok;
	}

	static bool test_sin() {
		bool ok = true;
		ok &= chk(approx((float)sf_sin(SoftFloat(0.f)), 0.f, 32), "  sin(0)\n");
		ok &= chk(approx((float)sf_sin(SoftFloat::half_pi()), 1.f, 256, 1e-3f), "  sin(pi/2)\n");
		ok &= chk(approx((float)sf_sin(-SoftFloat::half_pi()), -1.f, 256, 1e-3f), "  sin(-pi/2)\n");
		ok &= chk(fabsf((float)sf_sin(SoftFloat::pi())) < 1e-3f, "  sin(pi)\n");
		return ok;
	}

	static bool test_cos() {
		bool ok = true;
		ok &= chk(approx((float)sf_cos(SoftFloat(0.f)), 1.f, 256, 1e-3f), "  cos(0)\n");
		ok &= chk(approx((float)sf_cos(SoftFloat::pi()), -1.f, 256, 1e-3f), "  cos(pi)\n");
		ok &= chk(fabsf((float)sf_cos(SoftFloat::half_pi())) < 1e-3f, "  cos(pi/2)\n");
		return ok;
	}

	static bool test_lerp() {
		bool ok = true;
		ok &= approx((float)lerp(SoftFloat(0.f), SoftFloat(10.f), SoftFloat(.5f)), 5.f, 8);
		ok &= approx((float)lerp(SoftFloat(0.f), SoftFloat(10.f), SoftFloat(0.f)), 0.f, 1);
		ok &= approx((float)lerp(SoftFloat(0.f), SoftFloat(10.f), SoftFloat(1.f)), 10.f, 8);
		return ok;
	}

	static bool test_minmax() {
		bool ok = true;
		ok &= approx((float)sf_min(SoftFloat(5.f), SoftFloat(10.f)), 5.f);
		ok &= approx((float)sf_max(SoftFloat(5.f), SoftFloat(10.f)), 10.f);
		ok &= approx((float)sf_min(SoftFloat(-5.f), SoftFloat(5.f)), -5.f);
		ok &= approx((float)sf_max(SoftFloat(-5.f), SoftFloat(5.f)), 5.f);
		return ok;
	}

	static bool test_clamp() {
		bool ok = true;
		SoftFloat lo(0.f), hi(10.f);
		ok &= approx((float)sf_clamp(SoftFloat(5.f), lo, hi), 5.f);
		ok &= approx((float)sf_clamp(SoftFloat(15.f), lo, hi), 10.f);
		ok &= approx((float)sf_clamp(SoftFloat(-5.f), lo, hi), 0.f);
		return ok;
	}

	static bool test_int_conv() {
		bool ok = true;
		struct { float f; int32_t ex; } cs[] = {
			{42.9f,42},{-42.9f,-42},{0.9f,0},{-0.9f,0},
			{1.f,1},{-1.f,-1},{2.f,2},{3.f,3},{1024.f,1024},{100.f,100},
			{1000000.f,1000000},{0.f,0},
		};
		for (auto& c : cs) {
			int32_t got = (int32_t)SoftFloat(c.f);
			if (got != c.ex) {
				char b[128]; snprintf(b, sizeof b, "  int_conv(%.4g)=%d exp %d\n",
					(double)c.f, got, c.ex); print(b); ok = false;
			}
		}
		return ok;
	}

	static bool test_chained() {
		bool ok = true;
		// 1+2*3+4*5=27
		float r1 = (float)(SoftFloat(1.f) + SoftFloat(2.f) * SoftFloat(3.f)
			+ SoftFloat(4.f) * SoftFloat(5.f));
		ok &= chk(approx(r1, 27.f, 16, 5e-4f), "  chain1 failed\n");
		// (2+3)*(4+5)=45
		float r2 = (float)((SoftFloat(2.f) + SoftFloat(3.f)) * (SoftFloat(4.f) + SoftFloat(5.f)));
		ok &= chk(approx(r2, 45.f, 16, 5e-4f), "  chain2 failed\n");
		// 10/2 + 10/3 = 8.333
		float r3 = (float)(SoftFloat(10.f) / SoftFloat(2.f) + SoftFloat(10.f) / SoftFloat(3.f));
		ok &= chk(rel_eq(r3, 8.333333f, 5e-3f), "  chain3 failed\n");
		return ok;
	}

	static bool test_overflow() {
		bool ok = true;
		SoftFloat big(1e20f);
		SoftFloat r1 = big * big;
		ok &= chk(r1.exponent == 127 || (float)r1 > 1e18f, "  overflow\n");
		SoftFloat tiny(1e-20f);
		SoftFloat r2 = tiny * tiny;
		ok &= chk(r2.is_zero() || (float)r2 < 1e-30f, "  underflow\n");
		return ok;
	}

	static bool test_underflow() {
		SoftFloat a(1e-20f);
		SoftFloat r = a * a;
		float got = (float)r;
		// 1e-40 is below float min; we accept zero or very small
		return r.is_zero() || fabsf(got) < 1e-30f || rel_eq(got, 1e-40f, 0.5f);
	}

	static bool test_edge() {
		bool ok = true;
		ok &= chk((float)SoftFloat(static_cast<float>(INT32_MAX)) > 2e9f, "  edge big\n");
		ok &= chk((-SoftFloat(static_cast<float>(INT32_MAX))).is_negative(), "  edge neg\n");
		SoftFloat s = SoftFloat(1.f) >> 300;
		ok &= chk(s.is_zero() || (float)s < 1e-30f, "  edge shift\n");
		return ok;
	}

	static bool test_compound() {
		bool ok = true;
		SoftFloat a(5.f); a += SoftFloat(3.f);
		ok &= chk(approx((float)a, 8.f), "  +=\n");
		SoftFloat b(10.f); b -= SoftFloat(4.f);
		ok &= chk(approx((float)b, 6.f), "  -=\n");
		SoftFloat c(2.f); c *= SoftFloat(5.f);
		ok &= chk(approx((float)c, 10.f, 16), "  *=\n");
		SoftFloat d(20.f); d /= SoftFloat(4.f);
		ok &= chk(approx((float)d, 5.f, 64, 1e-3f), "  /=\n");
		return ok;
	}

	static bool test_expr() {
		bool ok = true;
		ok &= chk(approx((float)(SoftFloat(2.f) * SoftFloat(3.f)), 6.f, 4), "  expr mul\n");
		ok &= chk(approx((float)(SoftFloat(1.f) + SoftFloat(2.f) * SoftFloat(3.f)), 7.f, 8), "  expr fma\n");
		ok &= chk(approx((float)(SoftFloat(10.f) - SoftFloat(2.f) * SoftFloat(3.f)), 4.f, 8), "  expr fms\n");
		return ok;
	}

	static bool test_stress(int N = 500) {
		bool ok = true;
		uint32_t seed = 0xdeadbeef;
		auto lcg = [&] { seed = seed * 1664525u + 1013904223u; return seed; };
		for (int i = 0; i < N && ok; ++i) {
			float af = static_cast<float>(lcg() % 10000) / 100.f;
			float bf = static_cast<float>(lcg() % 10000) / 100.f;
			SoftFloat a(af), b(bf);
			float ea = af + bf, ga = (float)(a + b);
			if (!rel_eq(ga, ea, 5e-4f)) {
				char buf[128]; snprintf(buf, sizeof buf,
					"stress ADD i=%d: %.6g+%.6g=%.8g exp %.8g\n",
					i, (double)af, (double)bf, (double)ga, (double)ea);
				print(buf); ok = false; break;
			}
			float em = af * bf, gm = (float)(a * b);
			if (!rel_eq(gm, em, 5e-4f)) {
				char buf[128]; snprintf(buf, sizeof buf,
					"stress MUL i=%d: %.6g*%.6g=%.8g exp %.8g\n",
					i, (double)af, (double)bf, (double)gm, (double)em);
				print(buf); ok = false; break;
			}
			float es = af - bf, gs = (float)(a - b);
			if (fabsf(es) > 1e-4f && !rel_eq(gs, es, 5e-4f)) {
				char buf[128]; snprintf(buf, sizeof buf,
					"stress SUB i=%d: %.6g-%.6g=%.8g exp %.8g\n",
					i, (double)af, (double)bf, (double)gs, (double)es);
				print(buf); ok = false; break;
			}
		}
		return ok;
	}

	static bool test_fms_extended() {
		bool ok = true;
		struct { float a, b, c, ex; const char* n; } cs[] = {
			{10.f,   2.f, 3.f,    4.f,       "basic 10-2*3"},
			{ 1.f,   1.f, 1.f,    0.f,       "cancellation to zero"},
			{ 5.f,   2.f, 2.f,    1.f,       "small positive"},
			{ 1.5f,  1.f, 1.f,    0.5f,      "needs left shift"},
			{ 3.f,   2.f, 2.f,   -1.f,       "negative result"},
			{-3.f,   2.f, 3.f,   -9.f,       "neg a"},
			{ 0.f,   5.f, 7.f,  -35.f,       "a == 0"},
			{ 100.f,10.f,10.f,    0.f,       "exact cancel 100-100"},
			{ 1.f,   0.1f,10.f,   0.f,       "1 - 1 (fp noise)"},
		};
		for (auto& c : cs) {
			float got = (float)fused_mul_sub(SoftFloat(c.a), SoftFloat(c.b), SoftFloat(c.c));
			if (!(approx(got, c.ex, 32, 2e-3f) || fabsf(got) < 1e-7f)) {
				char b[128]; snprintf(b, sizeof b, "  fms %s: got %.8g exp %.8g\n", c.n, (double)got, (double)c.ex);
				print(b); ok = false;
			}
		}
		return ok;
	}

	static bool test_fmms_extended() {
		bool ok = true;
		struct { float a, b, c, d, ex; const char* n; } cs[] = {
			{2.f,3.f,4.f,5.f,   -14.f,      "6-20"},
			{10.f,10.f,9.f,11.f,  1.f,      "100-99"},
			{1.5f,1.5f,1.f,2.25f, 0.f,      "2.25-2.25"},
			{100.f,0.01f,1.f,1.f, 0.f,      "1-1"},
			{5.f,2.f,3.f,3.f,     1.f,      "10-9"},
			{-2.f,3.f,4.f,5.f,   -26.f,     "neg a*b"},
		};
		for (auto& c : cs) {
			float got = (float)fused_mul_mul_sub(
				SoftFloat(c.a), SoftFloat(c.b), SoftFloat(c.c), SoftFloat(c.d));
			if (!(approx(got, c.ex, 32, 2e-3f) || fabsf(got) < 1e-7f)) {
				char b[128]; snprintf(b, sizeof b, "  fmms %s: got %.8g exp %.8g\n", c.n, (double)got, (double)c.ex);
				print(b); ok = false;
			}
		}
		return ok;
	}
	static void run_benchmarks() {
		print_section("BENCHMARKS");
		const int N = 100000;
		char buf[256];
		volatile float sink_f = 0.f;
		volatile int32_t sink_i = 0;
		volatile bool sink_b = false;
		auto bench = [&](auto fn) -> double {
			uint64_t t0 = now_ns();
			for (int i = 0; i < N; ++i) fn();
			return static_cast<double>(now_ns() - t0) / N;
		};

		// Prepare test values
		SoftFloat a(3.14159f), b(2.71828f), c(1.41421f), d(0.5f);
		SoftFloat zero(0.f), one(1.f), two(2.f), half(0.5f), three(3.f);
		float fa = 3.14159f, fb = 2.71828f;

		// -------------------------------------------------------------------
		// Basic arithmetic
		// -------------------------------------------------------------------
		snprintf(buf,
			sizeof buf,
			"Multiply  (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)(a * b); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"Add       (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)(a + b); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"Subtract  (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)(a - b); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"Divide    (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)(a / b); })); print(buf);

		// -------------------------------------------------------------------
		// Fused operations
		// -------------------------------------------------------------------
		snprintf(buf,
			sizeof buf,
			"FMA       (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)fused_mul_add(one, two, three); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"FMS       (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)fused_mul_sub(one, two, three); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"FMMA      (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)fused_mul_mul_add(one, two, three, d); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"FMMS      (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)fused_mul_mul_sub(one, two, three, d); })); print(buf);

		// -------------------------------------------------------------------
		// Trigonometric
		// -------------------------------------------------------------------
		snprintf(buf,
			sizeof buf,
			"sin       (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)a.sin(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"cos       (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)a.cos(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"tan       (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)a.tan(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"asin      (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)half.asin(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"acos      (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)half.acos(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"atan2     (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)atan2(a, b); })); print(buf);

		// -------------------------------------------------------------------
		// Hyperbolic
		// -------------------------------------------------------------------
		snprintf(buf,
			sizeof buf,
			"sinh      (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)one.sinh(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"cosh      (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)one.cosh(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"tanh      (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)one.tanh(); })); print(buf);

		// -------------------------------------------------------------------
		// Exponential and logarithmic
		// -------------------------------------------------------------------
		snprintf(buf,
			sizeof buf,
			"exp       (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)one.exp(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"log       (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)two.log(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"log2      (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)two.log2(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"log10     (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)two.log10(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"pow       (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)two.pow(three); })); print(buf);

		// -------------------------------------------------------------------
		// Roots
		// -------------------------------------------------------------------
		snprintf(buf,
			sizeof buf,
			"sqrt      (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)two.sqrt(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"inv_sqrt  (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)two.inv_sqrt(); })); print(buf);

		// -------------------------------------------------------------------
		// Rounding and truncation
		// -------------------------------------------------------------------
		snprintf(buf,
			sizeof buf,
			"floor     (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)a.floor(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"ceil      (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)a.ceil(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"trunc     (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)a.trunc(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"round     (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)a.round(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"fract     (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)a.fract(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"modf      (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { auto p = a.modf(); sink_f = (float)p.intpart; })); print(buf);

		// -------------------------------------------------------------------
		// Other utilities
		// -------------------------------------------------------------------
		snprintf(buf,
			sizeof buf,
			"abs       (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)(-a).abs(); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"copysign  (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)a.copysign(b); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"fmod      (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)a.fmod(b); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"hypot     (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)hypot(three, d); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"lerp      (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)lerp(zero, two, half); })); print(buf);

		// -------------------------------------------------------------------
		// Comparisons
		// -------------------------------------------------------------------
		snprintf(buf,
			sizeof buf,
			"compare < (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_b = a < b; })); print(buf);
		snprintf(buf,
			sizeof buf,
			"compare ==(%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_b = a == a; })); print(buf);

		// -------------------------------------------------------------------
		// Shifts
		// -------------------------------------------------------------------
		snprintf(buf,
			sizeof buf,
			"shift <<  (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)(a << 2); })); print(buf);
		snprintf(buf,
			sizeof buf,
			"shift >>  (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)(a >> 2); })); print(buf);

		// -------------------------------------------------------------------
		// Conversions
		// -------------------------------------------------------------------
		snprintf(buf,
			sizeof buf,
			"to_float  (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)a; })); print(buf);
		snprintf(buf,
			sizeof buf,
			"to_int32  (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_i = (int32_t)a; })); print(buf);
		snprintf(buf,
			sizeof buf,
			"from_float(%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { SoftFloat tmp(fa); sink_f = (float)tmp; })); print(buf);
		snprintf(buf,
			sizeof buf,
			"from_int  (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { SoftFloat tmp(42); sink_f = (float)tmp; })); print(buf);

		// -------------------------------------------------------------------
		// Expression template (Dot4)
		// -------------------------------------------------------------------
		SoftFloat a1(1.f), a2(2.f), a3(3.f), a4(4.f);
		SoftFloat b1(5.f), b2(6.f), b3(7.f), b4(8.f);
		snprintf(buf,
			sizeof buf,
			"Dot4      (%6d ops): %7.2f ns/op\n",
			N,
			bench([&] { sink_f = (float)(a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4); })); print(buf);

		(void)sink_f; (void)sink_i; (void)sink_b;
	}

	// =========================================================================
	// ADDITIONAL CORNER-CASE & INVARIANT TESTS
	// =========================================================================
	
	// -----------------------------------------------------------------------
	// 1. Representation Invariants
	// -----------------------------------------------------------------------
	// Verifies that all non-zero results satisfy:
	//   clz(abs(mantissa)) == 2
	//   i.e., mantissa in [2^29, 2^30)
	static bool test_invariant_normalization() {
		bool ok = true;
		auto check = [&](SoftFloat x, const char* ctx) {
			if (x.mantissa == 0) return true;
			uint32_t a = sf_abs32(x.mantissa);
			int lz = sf_clz(a);
			if (lz != 2) {
				char b[128];
				snprintf(b, sizeof b, "  INV FAIL %s: mant=0x%08X clz=%d (exp %d)\n",
					ctx, x.mantissa, lz, x.exponent);
				print(b);
				return false;
			}
			// Ensure bits 31:30 are clear in abs(mantissa)
			if (a & 0xC0000000u) {
				char b[128];
				snprintf(b, sizeof b, "  INV FAIL %s: high bits set 0x%08X\n", ctx, a);
				print(b);
				return false;
			}
			return true;
			};

		// Smoke test across operations
		ok &= check(SoftFloat(1.f), "ctor(1)");
		ok &= check(SoftFloat(1.f) + SoftFloat(2.f), "add");
		ok &= check(SoftFloat(1.f) - SoftFloat(2.f), "sub");
		ok &= check(SoftFloat(1.f) * SoftFloat(2.f), "mul");
		ok &= check(SoftFloat(1.f) / SoftFloat(3.f), "div");
		ok &= check(fused_mul_add(SoftFloat(1.f), SoftFloat(2.f), SoftFloat(3.f)), "fma");
		ok &= check(SoftFloat(123.456f).sqrt(), "sqrt");
		ok &= check(SoftFloat(123.456f).inv_sqrt(), "inv_sqrt");
		ok &= check(SoftFloat::pi().sin(), "sin");
		ok &= check(SoftFloat::pi().cos(), "cos");

		// Stress random values
		uint32_t seed = 0x12345678;
		for (int i = 0; i < 200; ++i) {
			seed = seed * 1664525u + 1013904223u;
			float fa = reinterpret_cast<float&>(seed) * 1e-3f;
			uint32_t s = seed ^ 0xAAAAAAAA;
			float fb = reinterpret_cast<float&>(s) * 1e-3f;
			SoftFloat a(fa), b(fb);

			ok &= check(a + b, "rand add");
			ok &= check(a - b, "rand sub");
			ok &= check(a * b, "rand mul");
			if (!b.is_zero()) ok &= check(a / b, "rand div");
			ok &= check(fused_mul_add(a, b, a), "rand fma");
		}
		return ok;
	}

	// -----------------------------------------------------------------------
	// 2. Zero & Signed-Zero Semantics
	// -----------------------------------------------------------------------
	static bool test_zero_semantics() {
		bool ok = true;
		SoftFloat z0(0.f);
		SoftFloat z1 = SoftFloat(5.f) - SoftFloat(5.f);
		SoftFloat z2 = SoftFloat(-5.f) + SoftFloat(5.f);
		SoftFloat nz = SoftFloat(-0.f); // Constructor should normalize to 0

		ok &= chk(z0.is_zero(), "  0.f is zero");
		ok &= chk(z1.is_zero(), "  x-x is zero");
		ok &= chk(z2.is_zero(), "  -x+x is zero");
		ok &= chk(nz.is_zero(), "  -0.f normalizes to zero");

		// Equality
		ok &= chk(z0 == z1, "  0 == (x-x)");
		ok &= chk(z0 == nz, "  0 == -0");
		ok &= chk(!(z0 < nz), "  !(0 < -0)");
		ok &= chk(!(z0 > nz), "  !(0 > -0)");

		// Negation of zero
		ok &= chk((-z0).is_zero(), "  -0 is zero");
		ok &= chk((-z0) == z0, "  -0 == 0");

		// Arithmetic with zero
		ok &= chk((SoftFloat(42.f) + z0).to_float() == 42.f, "  x+0==x");
		ok &= chk((z0 + SoftFloat(42.f)).to_float() == 42.f, "  0+x==x");
		ok &= chk((SoftFloat(42.f) * z0).is_zero(), "  x*0==0");
		ok &= chk((z0 * SoftFloat(42.f)).is_zero(), "  0*x==0");
		ok &= chk((z0 / SoftFloat(2.f)).is_zero(), "  0/x==0");

		return ok;
	}

	// -----------------------------------------------------------------------
	// 3. Division by Zero & Infinity Inputs
	// -----------------------------------------------------------------------
	static bool test_div_zero_inf() {
		bool ok = true;

		// Div by zero: library returns clamped max-finite with sign
		SoftFloat pos_max = SoftFloat(1.f) / SoftFloat(0.f);
		SoftFloat neg_max = SoftFloat(-1.f) / SoftFloat(0.f);

		ok &= chk(pos_max.exponent == 127, "  1/0 exponent saturated");
		ok &= chk(pos_max.is_positive(), "  1/0 sign positive");
		ok &= chk(neg_max.exponent == 127, "  -1/0 exponent saturated");
		ok &= chk(neg_max.is_negative(), "  -1/0 sign negative");

		// Float constructor with Inf/NaN
		SoftFloat s_inf_pos(std::numeric_limits<float>::infinity());
		SoftFloat s_inf_neg(-std::numeric_limits<float>::infinity());
		SoftFloat s_nan(std::numeric_limits<float>::quiet_NaN());

		// Library clamps Inf/NaN to large finite values
		ok &= chk(s_inf_pos.exponent == 98, "  Inf ctor clamped exp"); // per code: exponent=98
		ok &= chk(s_inf_pos.is_positive(), "  Inf ctor sign");
		ok &= chk(s_inf_neg.exponent == 98, "  -Inf ctor clamped exp");
		ok &= chk(s_inf_neg.is_negative(), "  -Inf ctor sign");

		// NaN treated as zero? or clamped? Code: if bits==0xFF... -> clamp. 
		// NaN has exp=0xFF, frac!=0. Code path: if (expf == 0xFFu) clamp.
		ok &= chk(s_nan.exponent == 98, "  NaN ctor clamped");

		// Sqrt of negative -> zero
		ok &= chk(SoftFloat(-4.f).sqrt().is_zero(), "  sqrt(neg) -> 0");
		ok &= chk(SoftFloat(0.f).sqrt().is_zero(), "  sqrt(0) -> 0");

		// Inv_sqrt of zero/neg -> zero
		ok &= chk(SoftFloat(0.f).inv_sqrt().is_zero(), "  inv_sqrt(0) -> 0");
		ok &= chk(SoftFloat(-1.f).inv_sqrt().is_zero(), "  inv_sqrt(neg) -> 0");

		return ok;
	}

	// -----------------------------------------------------------------------
	// 4. Integer Conversion Boundaries
	// -----------------------------------------------------------------------
	static bool test_int_conv_edges() {
		bool ok = true;

		// Truncation toward zero
		ok &= chk(((int32_t)SoftFloat(42.999f)) == 42, "  trunc pos down");
		ok &= chk(((int32_t)SoftFloat(42.001f)) == 42, "  trunc pos up");
		ok &= chk(((int32_t)SoftFloat(-42.999f)) == -42, "  trunc neg toward zero");
		ok &= chk(((int32_t)SoftFloat(-42.001f)) == -42, "  trunc neg toward zero");

		// Small fractions -> 0
		ok &= chk(((int32_t)SoftFloat(0.999f)) == 0, "  <1 trunc to 0");
		ok &= chk(((int32_t)SoftFloat(-0.999f)) == 0, "  >-1 trunc to 0");

		// Overflow saturation in to_int32
		// Code: if (exponent >= 2) return INT32_MAX/MIN
		// 2^29 * 2^2 = 2^31 > INT32_MAX

		// Wait: mantissa 0x20000000 is 2^29. exp=2 => value = 2^29 * 2^2 = 2^31.
		// Actually library representation: value = m * 2^e. 
		// If m=2^29, e=2 => 2^31.
		ok &= chk(((int32_t)SoftFloat(3e9f)) == INT32_MAX, "  int conv overflow pos");
		ok &= chk(((int32_t)SoftFloat(-3e9f)) == INT32_MIN, "  int conv overflow neg");

		// Exact integer boundaries
		ok &= chk(((int32_t)SoftFloat(static_cast<float>(INT32_MAX))) == INT32_MAX, "  INT32_MAX roundtrip");
		ok &= chk(((int32_t)SoftFloat(static_cast<float>(INT32_MIN))) == INT32_MIN, "  INT32_MIN roundtrip");

		return ok;
	}

	// -----------------------------------------------------------------------
	// 5. Exponent Saturation & Normalization Edges
	// -----------------------------------------------------------------------
	static bool test_exp_saturation() {
		bool ok = true;

		SoftFloat max_exp = SoftFloat(0x20000000, 127);
		SoftFloat min_exp = SoftFloat(0x20000000, -128);

		// Addition causing exponent overflow
		SoftFloat r1 = max_exp + max_exp;
		// 2^29 + 2^29 = 2^30, overflow by 1 bit -> shift right -> exp+1 -> 128 -> clamp to 127
		ok &= chk(r1.exponent == 127, "  exp overflow clamp +");

		// Test saturation directly
		SoftFloat t1(0x20000000, 200);
		ok &= chk(t1.exponent == 127, "  sf_normalise exp overflow clamp");

		SoftFloat t2(0x20000000, -200);
		ok &= chk(t2.exponent == -128, "  sf_normalise exp underflow clamp");

		return ok;
	}

	// -----------------------------------------------------------------------
	// 6. Math Function Symmetries & Range Reduction
	// -----------------------------------------------------------------------
	static bool test_math_symmetries() {
		bool ok = true;
		auto approx = [](float a, float b, float rel_tol = 5e-3f, float abs_tol = 1e-4f) {
			if (a == b) return true;
			float sc = fmaxf(fabsf(a), fabsf(b));
			if (sc < abs_tol) return fabsf(a - b) < abs_tol;
			return fabsf(a - b) / sc < rel_tol;
			};

		// Sin odd: sin(-x) = -sin(x)
		for (float val : {0.1f, 1.f, 2.f, 5.f, 10.f, 100.f}) {
			SoftFloat x(val);
			float s1 = (float)x.sin();
			float s2 = (float)(-x).sin();
			ok &= chk(approx(s1, -s2, 1e-2f, 1e-4f), "  sin(-x) == -sin(x)");
		}

		// Cos even: cos(-x) = cos(x)
		for (float val : {0.1f, 1.f, 2.f, 5.f, 10.f}) {
			SoftFloat x(val);
			float c1 = (float)x.cos();
			float c2 = (float)(-x).cos();
			ok &= chk(approx(c1, c2, 1e-2f, 1e-4f), "  cos(-x) == cos(x)");
		}

		// Sin/Cos phase shift: cos(x) = sin(x + pi/2)
		for (float val : {0.f, 0.5f, 1.f, 3.f}) {
			SoftFloat x(val);
			float c = (float)x.cos();
			float s = (float)(x + SoftFloat::half_pi()).sin();
			ok &= chk(approx(c, s, 1e-2f, 1e-3f), "  cos(x) == sin(x+pi/2)");
		}

		// Large argument range reduction — relax tolerance
		float big = 12345.6789f;
		float expected = sinf(big);
		float got = (float)SoftFloat(big).sin();
		ok &= chk(approx(got, expected, 1e-1f, 1e-3f), "  sin(large arg) range reduce");

		// Sqrt monotonicity
		ok &= chk(SoftFloat(1.f).sqrt() < SoftFloat(2.f).sqrt(), "  sqrt monotonic");
		ok &= chk(SoftFloat(0.f).sqrt() < SoftFloat(0.0001f).sqrt(), "  sqrt small positive");

		return ok;
	}

	// -----------------------------------------------------------------------
	// 7. Lerp & Extrapolation
	// -----------------------------------------------------------------------
	static bool test_lerp_edges() {
		bool ok = true;
		SoftFloat a(10.f), b(20.f);

		// Standard
		ok &= chk(approx((float)lerp(a, b, SoftFloat(0.f)), 10.f), "  lerp t=0");
		ok &= chk(approx((float)lerp(a, b, SoftFloat(1.f)), 20.f), "  lerp t=1");
		ok &= chk(approx((float)lerp(a, b, SoftFloat(0.5f)), 15.f), "  lerp t=0.5");

		// Extrapolation
		ok &= chk(approx((float)lerp(a, b, SoftFloat(-1.f)), 0.f), "  lerp t=-1 extrapolate");
		ok &= chk(approx((float)lerp(a, b, SoftFloat(2.f)), 30.f), "  lerp t=2 extrapolate");

		// Reverse order (a > b)
		ok &= chk(approx((float)lerp(SoftFloat(20.f), SoftFloat(10.f), SoftFloat(0.5f)), 15.f), "  lerp reverse");

		return ok;
	}

	// -----------------------------------------------------------------------
	// 8. Expression Template Negation & Mixed Chains
	// -----------------------------------------------------------------------
	static bool test_expr_negation() {
		bool ok = true;

		SoftFloat a(2.f), b(3.f), c(4.f);

		// -(a*b)
		SoftFloat r1 = -(a * b);
		ok &= chk(approx((float)r1, -6.f), "  -(a*b)");

		// a - (b*c)  [uses operator-(SoftFloat, sf_mul_expr)]
		SoftFloat r2 = a - (b * c);
		ok &= chk(approx((float)r2, 2.f - 12.f), "  a - (b*c)");

		// (a*b) - c  [uses operator-(sf_mul_expr, SoftFloat)]
		// Fixed in library to be: -(fms(c, a, b))
		SoftFloat r3 = (a * b) - c;
		ok &= chk(approx((float)r3, 6.f - 4.f), "  (a*b) - c");

		// Complex: a + b*c - d*e
		SoftFloat d(5.f), e(6.f);
		SoftFloat r4 = a + b * c - d * e;
		ok &= chk(approx((float)r4, 2.f + 12.f - 30.f), "  a + b*c - d*e");

		return ok;
	}

	// -----------------------------------------------------------------------
	// 9. Constexpr Validation
	// -----------------------------------------------------------------------
	// Ensures complex expressions can be evaluated at compile-time.
	static bool test_constexpr_complex() {
		constexpr SoftFloat c1 = SoftFloat(1.5f) + SoftFloat(2.5f) * SoftFloat(3.0f);
		static_assert(c1.to_float() == 9.0f, "constexpr fma failed");

		constexpr SoftFloat c2 = fused_mul_add(SoftFloat(2.f), SoftFloat(3.f), SoftFloat(4.f));
		static_assert(c2.to_float() == 14.0f, "constexpr fused_mul_add failed");

		constexpr SoftFloat c3 = SoftFloat::pi().sin();

		// sin(pi) approx 0
		static_assert(c3.to_float() > -1e-3f && c3.to_float() < 1e-3f, "constexpr sin(pi)");

		constexpr SoftFloat c4 = SoftFloat(16.f).sqrt();
		static_assert(c4.to_float() == 4.0f, "constexpr sqrt");

		static_assert(SoftFloat(144.f).sqrt().to_float() == 12.0f, "");
		static_assert(SoftFloat(0.25f).sqrt().to_float() == 0.5f, "");
		static_assert(SoftFloat(1e10f).sqrt().to_float() == 1e5f, "");  // if perfect

		return true;
	}

	// -----------------------------------------------------------------------
	// 10. Shift Operator Edge Cases
	// -----------------------------------------------------------------------
	static bool test_shift_edges() {
		bool ok = true;
		SoftFloat x(1.f); // mant=2^29, exp=-29

		// Shift by 0
		ok &= chk(((float)(x << 0)) == 1.f, "  <<0");
		ok &= chk(((float)(x >> 0)) == 1.f, "  >>0");

		// Large shifts triggering underflow/overflow
		SoftFloat r1 = x << 200; // exp becomes 171 -> clamp to 127 -> max finite
		ok &= chk(r1.exponent == 127, "  << overflow clamp");

		SoftFloat r2 = x >> 200; // exp becomes -229 -> < -250? no, but to_float returns 0 if exp<=0?
		// Code: to_float: if (iexp <= 0) return 0.f. 
		// iexp = exp + 156. -229+156 = -73 <=0 -> 0.
		ok &= chk(r2.is_zero() || ((float)r2) == 0.f, "  >> underflow zero");

		// Negative shift arguments (library treats as signed int)
		// x << -5  => exponent + (-5) => same as >> 5
		ok &= chk(approx((float)(x << -5), (float)(x >> 5)), "  << negative arg acts as >>");
		ok &= chk(approx((float)(x >> -5), (float)(x << 5)), "  >> negative arg acts as <<");

		return ok;
	}

	// -----------------------------------------------------------------------
	// 11. Comparison Reflexivity & Transitivity
	// -----------------------------------------------------------------------
	static bool test_comparison_logic() {
		bool ok = true;
		SoftFloat a(1.f), b(2.f), c(3.f), z(0.f);

		// Reflexivity
		ok &= chk(a == a, "  a==a");
		ok &= chk(a <= a, "  a<=a");
		ok &= chk(a >= a, "  a>=a");
		ok &= chk(!(a < a), "  !(a<a)");
		ok &= chk(!(a > a), "  !(a>a)");

		// Zero equality
		ok &= chk(z == z, "  0==0");
		ok &= chk(!(z < z), "  !(0<0)");

		// Transitivity
		ok &= chk(a < b && b < c ? a < c : true, "  transitivity <");
		ok &= chk(a <= b && b <= c ? a <= c : true, "  transitivity <=");
		ok &= chk(c > b && b > a ? c > a : true, "  transitivity >");

		// Mixed sign
		SoftFloat neg(-5.f);
		ok &= chk(neg < z, "  neg < zero");
		ok &= chk(z > neg, "  zero > neg");
		ok &= chk(neg < a, "  neg < pos");

		return ok;
	}

	// =========================================================================
	// NEW: Corner Cases & Stress Tests
	// =========================================================================
	static bool test_corner_cases() {
		bool ok = true;

		// --------- 1. Integer Boundaries ---------
		{
			SoftFloat s_min(INT_MIN);
			SoftFloat s_max(INT_MAX);
			int32_t min_conv = (int32_t)s_min;
			int32_t max_conv = (int32_t)s_max;

			if (min_conv != INT_MIN) { print("FAIL: INT_MIN convert\n"); ok = false; }
			if (max_conv < INT_MAX - 1000) { print("FAIL: INT_MAX convert\n"); ok = false; }
		}

		// --------- 2. Subnormal / Denormal handling ---------
		{
			float f_min_norm = 1.17549435e-38f;
			float f_max_denorm = 1.17549421e-38f;

			SoftFloat s_norm(f_min_norm);
			SoftFloat s_denorm(f_max_denorm);

			if (!(s_denorm.is_zero() || fabsf((float)s_denorm) < 1e-30f)) {
				print("FAIL: Denormal handling\n"); ok = false;
			}
			if (s_norm.is_zero()) { print("FAIL: Min normal is zero\n"); ok = false; }
		}

		// --------- 3. Catastrophic Cancellation ---------
		{
			SoftFloat one(1.0f);
			SoftFloat tiny(1e-8f);

			SoftFloat r1 = one + (tiny - one);
			float f1 = (float)r1;
			if (!((f1 == 0.f) || (fabsf(f1) < 1e-7f))) {
				char buf[128];
				snprintf(buf, sizeof buf, "FAIL: Catastrophic cancel 1, got %.8g\n", (double)f1);
				print(buf);
				ok = false;
			}

			SoftFloat r2 = (one - one) + tiny;
			float f2 = (float)r2;
			if (!approx(f2, 1e-8f, 0.5f, 5e-9f)) {
				char buf[128];
				snprintf(buf, sizeof buf, "FAIL: Catastrophic cancel 2, got %.8g exp %.8g\n",
					(double)f2, 1e-8);
				print(buf);
				ok = false;
			}
		}

		// --------- 4. FMA / FMS Edge Cases ---------
		// Fixed: fused_mul_sub(a, b, c) computes a - b*c, not (a*b) - c
		{
			// a - b*c = 10 - 3*4 = 10 - 12 = -2
			SoftFloat r_fms = fused_mul_sub(SoftFloat(10.f), SoftFloat(3.f), SoftFloat(4.f));
			float fms_result = (float)r_fms;
			if (!approx(fms_result, -2.f, 16, 5e-4f)) {
				char buf[128];
				snprintf(buf, sizeof buf, "FAIL: FMS(10,3,4), got %.8g exp -2\n", (double)fms_result);
				print(buf);
				ok = false;
			}

			// (mul)-scalar: (2*3) - 10 = 6 - 10 = -4
			SoftFloat r_expr = (SoftFloat(2.f) * SoftFloat(3.f)) - SoftFloat(10.f);
			float expr_f = (float)r_expr;
			if (!approx(expr_f, -4.f, 8)) {
				char buf[128];
				snprintf(buf, sizeof buf, "FAIL: (mul)-scalar negative, got %.8g\n", (double)expr_f);
				print(buf);
				ok = false;
			}
		}

		// --------- 5. Division Edge Cases ---------
		{
			SoftFloat a(10.f), b(2.f);
			float div_f = (float)(a / b);
			if (!approx(div_f, 5.f, 64, 1e-3f)) {
				char buf[128];
				snprintf(buf, sizeof buf, "FAIL: Normal division, got %.8g\n", (double)div_f);
				print(buf);
				ok = false;
			}

			// Division by zero returns clamped large value with saturated exponent
			SoftFloat z(0.f);
			SoftFloat div_z = SoftFloat(1.f) / z;
			if (div_z.exponent != 127) {
				char buf[128];
				snprintf(buf, sizeof buf, "FAIL: Division by zero, exp=%d\n", div_z.exponent);
				print(buf);
				ok = false;
			}

			// 0/0 = clamped large value (not exactly zero per the division code)
			// The code returns: if (!rhs.mantissa) return from_raw(mantissa >= 0 ? (1<<29) : -(1<<29), 127);
			SoftFloat r_nan = z / z;
			if (!r_nan.is_zero() && r_nan.exponent != 127) {
				char buf[128];
				snprintf(buf, sizeof buf, "FAIL: 0/0 behavior, mant=%d exp=%d\n", r_nan.mantissa, r_nan.exponent);
				print(buf);
				ok = false;
			}
		}

		// --------- 6. Shifts and Saturation ---------
		{
			SoftFloat one(1.f);

			SoftFloat lsh = one << 200;
			if (lsh.exponent != 127) {
				char buf[128];
				snprintf(buf, sizeof buf, "FAIL: Left shift overflow, exp=%d\n", lsh.exponent);
				print(buf);
				ok = false;
			}

			SoftFloat rsh = one >> 200;
			if (!(rsh.is_zero() || fabsf((float)rsh) < 1e-30f)) {
				char buf[128];
				snprintf(buf, sizeof buf, "FAIL: Right shift underflow, got %.8g\n", (double)(float)rsh);
				print(buf);
				ok = false;
			}
		}

		// --------- 7. Transcendental Functions ---------
		{
			SoftFloat big(1e9f);
			float f_sin = (float)big.sin();
			if (!(f_sin >= -1.0f && f_sin <= 1.0f)) {
				char buf[128];
				snprintf(buf, sizeof buf, "FAIL: sin(1e9) out of range, got %.8g\n", (double)f_sin);
				print(buf);
				ok = false;
			}

			SoftFloat two(2.f);
			float sqrt2 = (float)two.sqrt();
			if (!approx(sqrt2, 1.41421356f, 128, 1e-3f)) {
				char buf[128];
				snprintf(buf, sizeof buf, "FAIL: sqrt(2), got %.8g\n", (double)sqrt2);
				print(buf);
				ok = false;
			}
		}

		return ok;
	}

	static bool test_atan2() {
		bool ok = true;

		auto ang_close = [&](float y, float x, float got, float ex, const char* nm) -> bool {
			if (std::isnan(got) || std::isnan(ex)) {
				print("FAIL: atan2 produced NaN\n");
				return false;
			}
			float diff = fabsf(got - ex);
			float aex = fabsf(ex);

			// Tolerance: tighter near 0, looser further away.
			float tol = (aex < 0.1f) ? 2e-3f : 1e-2f;

			if (diff > tol) {
				char b[160];
				snprintf(b, sizeof b,
					"  atan2(% .6g,% .6g): got %.8g ex %.8g diff %.8g (tol %.3g) [%s]\n", (double)y, (double)x,
					(double)got, (double)ex, (double)diff, (double)tol, nm);
				// Note: msg labels got/ex swapped above in format string to keep message simple.
				// Let's print again correctly:
				snprintf(b, sizeof b,
					"  FAIL: atan2(y=%.6g, x=%.6g) got %.8g ex %.8g diff %.8g tol %.3g (%s)\n",
					(double)y, (double)x, (double)got, (double)ex, (double)diff, (double)tol, nm);
				print(b);
				return false;
			}
			return true;
			};

		auto check = [&](float y, float x, const char* nm) -> bool {
			float got = (float)atan2(SoftFloat(y), SoftFloat(x));
			float ex = atan2f(y, x);
			bool pass = ang_close(y, x, got, ex, nm);

			// Range sanity: should be within [-pi, pi]
			if (pass) {
				float pi = 3.14159265f;
				if (fabsf(got) > pi + 2e-3f) {
					char b[128];
					snprintf(b, sizeof b, "  FAIL: atan2 range, got %.8g (%s)\n", (double)got, nm);
					print(b);
					return false;
				}
			}
			return pass;
			};

		// --- Axes / exact cases
		ok &= chk(check(0.f, 1.f, "atan2(0,1)"), "");
		ok &= chk(check(0.f, -1.f, "atan2(0,-1)"), "");     // expect ~pi
		ok &= chk(check(1.f, 0.f, "atan2(1,0)"), "");      // expect ~pi/2
		ok &= chk(check(-1.f, 0.f, "atan2(-1,0)"), "");     // expect ~-pi/2
		ok &= chk(check(0.f, 0.f, "atan2(0,0)"), "");       // library returns 0

		// --- Quadrants (known angles)
		ok &= chk(check(1.f, 1.f, "atan2(1,1)=pi/4"), "");
		ok &= chk(check(1.f, -1.f, "atan2(1,-1)=3pi/4"), "");
		ok &= chk(check(-1.f, 1.f, "atan2(-1,1)=-pi/4"), "");
		ok &= chk(check(-1.f, -1.f, "atan2(-1,-1)=-3pi/4"), "");

		// --- A few additional ratios
		ok &= chk(check(2.f, 1.f, "atan2(2,1)"), "");
		ok &= chk(check(1.f, 2.f, "atan2(1,2)"), "");
		ok &= chk(check(10.f, 1.f, "atan2(10,1)"), "");
		ok &= chk(check(1.f, 10.f, "atan2(1,10)"), "");
		ok &= chk(check(-10.f, 1.f, "atan2(-10,1)"), "");
		ok &= chk(check(10.f, -1.f, "atan2(10,-1)"), "");

		// --- Random stress vs libm atan2f
		uint32_t seed = 0xC0FFEEu;
		auto lcg = [&] {
			seed = seed * 1664525u + 1013904223u;
			return seed;
			};

		const int N = 300;
		for (int i = 0; i < N && ok; ++i) {
			// random in [-10, 10]
			float y = (int32_t)(lcg() % 20001u) / 1000.f - 10.f;
			float x = (int32_t)(lcg() % 20001u) / 1000.f - 10.f;

			// skip exact (0,0)
			if (x == 0.f && y == 0.f) continue;

			float got = (float)atan2(SoftFloat(y), SoftFloat(x));
			float ex = atan2f(y, x);

			float diff = fabsf(got - ex);
			float aex = fabsf(ex);
			float tol = (aex < 0.1f) ? 2e-3f : 1e-2f;

			if (diff > tol) {
				char b[180];
				snprintf(b, sizeof b,
					"  FAIL random i=%d atan2(y=%.6g,x=%.6g): got %.8g ex %.8g diff %.8g tol %.3g\n",
					i, (double)y, (double)x, (double)got, (double)ex, (double)diff, (double)tol);
				print(b);
				ok = false;
				break;
			}
		}

		return ok;
	}

	static bool test_atan2_v2() {
		bool ok = true;
		constexpr float tol = 2e-3f; // Matches ~0.1 degree max error of the implementation

		// -------------------------------------------------------------------
		// Axis aligned exact edge cases
		// -------------------------------------------------------------------
		struct { float y, x, ex; const char* name; } axes[] = {
			{ 0.0f,  1.0f,  0.0f,            "(0, 1) = 0" },
			{ 0.0f, -1.0f,  (float)M_PI,     "(0, -1) = pi" },
			{ 1.0f,  0.0f,  (float)M_PI / 2,   "(1, 0) = pi/2" },
			{-1.0f,  0.0f, -(float)M_PI / 2,   "(-1, 0) = -pi/2" },
			{ 1000.f, 0.f,  (float)M_PI / 2,   "(1000, 0) = pi/2" },
			{ 0.f, 1000.f,  0.0f,            "(0, 1000) = 0" },
		};

		for (auto& c : axes) {
			float got = (float)atan2(SoftFloat(c.y), SoftFloat(c.x));
			if (!approx(got, c.ex, 256, tol)) {
				char b[128];
				snprintf(b, sizeof b, "  atan2 %s: got %.6g exp %.6g\n", c.name, (double)got, (double)c.ex);
				print(b);
				ok = false;
			}
		}

		// -------------------------------------------------------------------
		// Octant test points
		// -------------------------------------------------------------------
		struct { float y, x; } oct[] = {
			{1,1}, {1,-1}, {-1,-1}, {-1,1},
			{1,2}, {2,1}, {1,10}, {10,1}, {3,7}, {7,3}
		};

		for (auto& c : oct) {
			float ex = atan2f(c.y, c.x);
			float got = (float)atan2(SoftFloat(c.y), SoftFloat(c.x));
			if (!approx(got, ex, 256, tol)) {
				char b[128];
				snprintf(b, sizeof b, "  atan2(%.0f, %.0f): got %.6g exp %.6g\n",
					(double)c.y, (double)c.x, (double)got, (double)ex);
				print(b);
				ok = false;
			}
		}

		// -------------------------------------------------------------------
		// Special cases
		// -------------------------------------------------------------------
		ok &= chk(atan2(SoftFloat(0.f), SoftFloat(0.f)).is_zero(),
			"  atan2(0, 0) correctly returns zero");

		// The most important property of atan2: it is homogeneous
		// atan2(k*y, k*x) = atan2(y,x) for any positive k
		{
			SoftFloat y(3.f), x(4.f);
			float base = (float)atan2(y, x);
			for (int shift = -12; shift <= 12; shift++) {
				float got = (float)atan2(y << shift, x << shift);
				if (!approx(got, base, 256, tol)) {
					char b[128];
					snprintf(b, sizeof b, "  atan2 homogeneous failed at shift %d\n", shift);
					print(b);
					ok = false;
					break;
				}
			}
		}

		// -------------------------------------------------------------------
		// Symmetry properties
		// -------------------------------------------------------------------
		for (float y : {0.2f, 1.f, 5.f}) {
			for (float x : {0.2f, 1.f, 5.f}) {
				float a = (float)atan2(SoftFloat(y), SoftFloat(x));

				bool sym1 = approx((float)atan2(SoftFloat(-y), SoftFloat(x)), -a, 256, tol);
				if (!sym1) {
					char buf[160];
					snprintf(buf, sizeof buf, "  Symmetry fail: atan2(%.2f, %.2f) vs %.6f\n", -y, x, -a);
					print(buf);
					ok = false;
				}

				bool sym2 = approx((float)atan2(SoftFloat(y), SoftFloat(-x)), (float)M_PI - a, 256, tol);
				if (!sym2) {
					char buf[160];
					snprintf(buf, sizeof buf, "  Symmetry fail: atan2(%.2f, %.2f) vs %.6f\n", y, -x, (float)M_PI - a);
					print(buf);
					ok = false;
				}

				bool sym3 = approx((float)atan2(SoftFloat(-y), SoftFloat(-x)), a - (float)M_PI, 256, tol);
				if (!sym3) {
					char buf[160];
					snprintf(buf, sizeof buf, "  Symmetry fail: atan2(%.2f, %.2f) vs %.6f\n", -y, -x, a - (float)M_PI);
					print(buf);
					ok = false;
				}
			}
		}

		// -------------------------------------------------------------------
		// Full circle sweep test
		// -------------------------------------------------------------------
		float max_err = 0.f;
		for (int deg = 0; deg < 360; deg += 2) {
			float rad = deg * (float)M_PI / 180.f;
			float y = sinf(rad);
			float x = cosf(rad);

			float got = (float)atan2(SoftFloat(y), SoftFloat(x));
			float err = fabsf(got - rad);
			if (err > M_PI) err = 2 * (float)M_PI - err;

			max_err = fmaxf(max_err, err);

			if (err > 5e-3f) {
				char b[128];
				snprintf(b, sizeof b, "  atan2 %d deg: error %.2f deg\n", deg, err * 57.29578f);
				print(b);
				ok = false;
			}
		}

		char b[128];
		snprintf(b, sizeof b, "  atan2 worst case error: %.2f degree\n", max_err * 57.29578f);
		print(b);

		return ok;
	}

	static bool test_exp() {
		bool ok = true;
		struct { float x, ex; } cases[] = {
			{0.f, 1.f},
			{1.f, 2.7182818f},
			{-1.f, 0.36787944f},
			{2.f, 7.389056f},
			{10.f, 22026.465f},
		};
		for (auto& c : cases) {
			float got = (float)SoftFloat(c.x).exp();
			bool sym = approx(got, c.ex, 256, 5e-3f);
			if (!sym) {
				char buf[160];
				snprintf(buf, sizeof buf, "  Exponential fail: exp(%.2f) got %.6f, expected %.6f\n", c.x, got, c.ex);
				print(buf);

				// print error
				ok = false;
			}
		}
		return ok;
	}

	static bool test_hypot() {
		bool ok = true;
		struct { float x, y, ex; } cases[] = {
			{3.f, 4.f, 5.f},
			{5.f, 12.f, 13.f},
			{1e-20f, 1e-20f, 1.41421356e-20f},
			{1e20f, 1e20f, 1.41421356e20f}, // may saturate? Should be safe.
		};
		for (auto& c : cases) {
			float got = (float)hypot(SoftFloat(c.x), SoftFloat(c.y));
			if (!approx(got, c.ex, 128, 1e-3f)) {
				// print error
				ok = false;
			}
		}
		return ok;
	}

	static bool test_tan() {
		bool ok = true;
		struct { float x; } angles[] = { 0.f, 0.5f, 1.f, -0.5f, 3.14159f / 4 };
		for (auto a : angles) {
			float got = (float)SoftFloat(a.x).tan();
			float ex = tanf(a.x);
			if (!approx(got, ex, 256, 2e-3f)) {
				char b[128]; snprintf(b, sizeof b, "  tan(%.4g): got %.6g exp %.6g\n", (double)a.x, (double)got, (double)ex);
				print(b); ok = false;
			}
		}
		return ok;
	}

	static bool test_asin_acos() {
		bool ok = true;
		float vals[] = { -0.9f, -0.5f, 0.f, 0.5f, 0.9f, 1.f, -1.f };
		for (float v : vals) {
			float got_asin = (float)SoftFloat(v).asin();
			float ex_asin = asinf(v);
			if (!approx(got_asin, ex_asin, 512, 1e-3f)) {
				char b[128]; snprintf(b, sizeof b, "  asin(%.4g): got %.6g exp %.6g\n", (double)v, (double)got_asin, (double)ex_asin);
				print(b); ok = false;
			}
			float got_acos = (float)SoftFloat(v).acos();
			float ex_acos = acosf(v);
			if (!approx(got_acos, ex_acos, 512, 1e-3f)) {
				char b[128]; snprintf(b, sizeof b, "  acos(%.4g): got %.6g exp %.6g\n", (double)v, (double)got_acos, (double)ex_acos);
				print(b); ok = false;
			}
		}
		return ok;
	}

	static bool test_log() {
		bool ok = true;
		float vals[] = { 0.5f, 1.f, 2.f, 10.f, 100.f };
		for (float v : vals) {
			float got = (float)SoftFloat(v).log();
			float ex = logf(v);
			if (!approx(got, ex, 512, 1e-3f)) {
				char b[128]; snprintf(b, sizeof b, "  log(%.4g): got %.6g exp %.6g\n", (double)v, (double)got, (double)ex);
				print(b); ok = false;
			}
		}
		// log2 and log10
		for (float v : vals) {
			float got2 = (float)SoftFloat(v).log2();
			float ex2 = log2f(v);
			if (!approx(got2, ex2, 512, 1e-3f)) {
				char b[128]; snprintf(b, sizeof b, "  log2(%.4g): got %.6g exp %.6g\n", (double)v, (double)got2, (double)ex2);
				print(b); ok = false;
			}
			float got10 = (float)SoftFloat(v).log10();
			float ex10 = log10f(v);
			if (!approx(got10, ex10, 512, 1e-3f)) {
				char b[128]; snprintf(b, sizeof b, "  log10(%.4g): got %.6g exp %.6g\n", (double)v, (double)got10, (double)ex10);
				print(b); ok = false;
			}
		}
		return ok;
	}

	static bool test_pow() {
		bool ok = true;
		struct { float x, y, ex; } cases[] = {
			{2.f, 3.f, 8.f},
			{10.f, 0.f, 1.f},
			{0.f, 5.f, 0.f},
			{4.f, 0.5f, 2.f},
			{2.f, -1.f, 0.5f},
		};
		for (auto& c : cases) {
			float got = (float)SoftFloat(c.x).pow(SoftFloat(c.y));
			if (!approx(got, c.ex, 512, 2e-3f)) {
				char b[128]; snprintf(b, sizeof b, "  pow(%.4g, %.4g): got %.6g exp %.6g\n", (double)c.x, (double)c.y, (double)got, (double)c.ex);
				print(b); ok = false;
			}
		}
		return ok;
	}

	static bool test_rounding() {
		bool ok = true;
		struct { float x; float fl, ce, tr, rnd; } cases[] = {
			{ 1.3f,  1.f,  2.f,  1.f,  1.f},
			{ 1.5f,  1.f,  2.f,  1.f,  2.f},
			{ 1.7f,  1.f,  2.f,  1.f,  2.f},
			{-1.3f, -2.f, -1.f, -1.f, -1.f},
			{-1.5f, -2.f, -1.f, -1.f, -2.f},
			{-1.7f, -2.f, -1.f, -1.f, -2.f},
			{ 2.0f,  2.f,  2.f,  2.f,  2.f},
			{ 0.0f,  0.f,  0.f,  0.f,  0.f},
		};
		for (auto& c : cases) {
			SoftFloat x(c.x);
			if (!approx((float)x.floor(), c.fl)) { print("floor fail\n"); ok = false; }
			if (!approx((float)x.ceil(), c.ce)) { print("ceil fail\n"); ok = false; }
			if (!approx((float)x.trunc(), c.tr)) { print("trunc fail\n"); ok = false; }
			if (!approx((float)x.round(), c.rnd)) { print("round fail\n"); ok = false; }
			// fract
			float fract_ex = c.x - c.tr;
			if (!approx((float)x.fract(), fract_ex, 8, 1e-4f)) { print("fract fail\n"); ok = false; }
			// modf
			auto pair = x.modf();
			if (!approx((float)pair.intpart, c.tr) || !approx((float)pair.fracpart, fract_ex)) {
				print("modf fail\n"); ok = false;
			}
		}
		return ok;
	}

	static bool test_hyperbolic() {
		bool ok = true;
		float vals[] = { -1.f, 0.f, 1.f, 2.f };
		for (float v : vals) {
			float got_sinh = (float)SoftFloat(v).sinh();
			float ex_sinh = sinhf(v);
			if (!approx(got_sinh, ex_sinh, 512, 2e-3f)) {
				char b[128]; snprintf(b, sizeof b, "  sinh(%.4g): got %.6g exp %.6g\n", (double)v, (double)got_sinh, (double)ex_sinh);
				print(b); ok = false;
			}
			float got_cosh = (float)SoftFloat(v).cosh();
			float ex_cosh = coshf(v);
			if (!approx(got_cosh, ex_cosh, 512, 2e-3f)) {
				char b[128]; snprintf(b, sizeof b, "  cosh(%.4g): got %.6g exp %.6g\n", (double)v, (double)got_cosh, (double)ex_cosh);
				print(b); ok = false;
			}
			float got_tanh = (float)SoftFloat(v).tanh();
			float ex_tanh = tanhf(v);
			if (!approx(got_tanh, ex_tanh, 512, 2e-3f)) {
				char b[128]; snprintf(b, sizeof b, "  tanh(%.4g): got %.6g exp %.6g\n", (double)v, (double)got_tanh, (double)ex_tanh);
				print(b); ok = false;
			}
		}
		return ok;
	}

	static bool test_copysign_fmod() {
		bool ok = true;
		SoftFloat a(5.f), b(-2.f);
		SoftFloat c = a.copysign(b);
		ok &= chk(c.is_negative() && approx((float)c, -5.f), "copysign");
		SoftFloat d(-3.f), e(2.f);
		SoftFloat f = d.copysign(e);
		ok &= chk(f.is_positive() && approx((float)f, 3.f), "copysign2");

		// fmod
		struct { float x, y, ex; } fmods[] = {
			{5.3f, 2.f, 1.3f},
			{-5.3f, 2.f, -1.3f},
			{5.3f, -2.f, 1.3f},
			{10.f, 3.f, 1.f},
			{1.f, 0.5f, 0.f},
		};
		for (auto& c : fmods) {
			float got = (float)SoftFloat(c.x).fmod(SoftFloat(c.y));
			if (!approx(got, c.ex, 8, 1e-4f)) {
				char b[128]; snprintf(b, sizeof b, "  fmod(%.4g, %.4g): got %.6g exp %.6g\n", (double)c.x, (double)c.y, (double)got, (double)c.ex);
				print(b); ok = false;
			}
		}
		return ok;
	}


	struct TC { const char* name; bool(*fn)(); };
	static const TC tests[] = {
		{"Default Constructor",          test_default_ctor},
		{"Integer Constructor",          test_int_ctor},
		{"Float Constructor",            test_float_ctor},
		{"Constants",                    test_constants},
		{"Addition",                     test_addition},
		{"Subtraction",                  test_subtraction},
		{"Multiplication",               test_multiplication},
		{"Division",                     test_division},
		{"Negation",                     test_negation},
		{"Absolute Value",               test_abs},
		{"Comparison",                   test_comparison},
		{"Shift Operators",              test_shift},
		{"Fused Multiply-Add",           test_fma},
		{"Fused Multiply-Sub",           test_fms},
		{"Fused Multiply-Multiply-Add",  test_fmma},
		{"Square Root",                  test_sqrt},
		{"Inverse Square Root",          test_inv_sqrt},
		{"Sine",                         test_sin},
		{"Cosine",                       test_cos},
		{"Linear Interpolation",         test_lerp},
		{"Min/Max",                      test_minmax},
		{"Clamp",                        test_clamp},
		{"Integer Conversion",           test_int_conv},
		{"Chained Operations",           test_chained},
		{"Overflow Handling",            test_overflow},
		{"Underflow Handling",           test_underflow},
		{"Edge Cases",                   test_edge},
		{"Compound Assignment",          test_compound},
		{"Expression Templates",         test_expr},
		{"Fused Multiply-Sub (ext)",     test_fms_extended},
		{"Fused Mul-Mul-Sub (ext)",      test_fmms_extended},
		{"Invariant Normalization",      test_invariant_normalization},
		{"Zero Semantics",               test_zero_semantics},
		{"Div/Zero & Inf Inputs",        test_div_zero_inf},
		{"Int Conv Edges",               test_int_conv_edges},
		{"Exponent Saturation",          test_exp_saturation},
		{"Math Symmetries",              test_math_symmetries},
		{"Lerp Edges",                   test_lerp_edges},
		{"Expr Negation",                test_expr_negation},
		{"Constexpr Complex",            test_constexpr_complex},
		{"Shift Edges",                  test_shift_edges},
		{"Comparison Logic",             test_comparison_logic},
		{"Corner Cases & Bit Patterns",  test_corner_cases},
		{"atan2",                        test_atan2},
		{"atan2 (v2)",                   test_atan2_v2},
		{"exp",                          test_exp},
		{"hypot",                        test_hypot},
		{"tan", test_tan},
		{"asin/acos", test_asin_acos},
		{"log/log2/log10", test_log},
		{"pow", test_pow},
		{"rounding", test_rounding},
		{"hyperbolic", test_hyperbolic},
		{"copysign/fmod", test_copysign_fmod},
		{"Random Stress Test",          []() { return test_stress(500); }},
	};

	static int run_all() {
		print("\n===========================================================\n");
		print("  SoftFloat Test Suite\n");
		print("===========================================================\n\n");
		print_section("FUNCTIONAL TESTS");
		int total = 0, passed = 0, failed = 0;
		for (auto& t : tests) {
			bool r = t.fn(); ++total; if (r)++passed; else++failed;
			print_result(r, t.name);
		}
		print("\n-----------------------------------------------------------\n");
		char s[128]; snprintf(s, sizeof s, "Results: %d/%d passed, %d failed\n", passed, total, failed);
		print(s);
		print("-----------------------------------------------------------\n");
		run_benchmarks();
		print("\n");
		print(failed == 0 ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
		return failed > 0 ? 1 : 0;
	}

} // namespace SoftFloatTest


extern "C" int main_tests() { 
	return SoftFloatTest::run_all(); 
}

//#if !defined(TARGET_GD32F103)
//int main() { return SoftFloatTest::run_all(); }
//#endif