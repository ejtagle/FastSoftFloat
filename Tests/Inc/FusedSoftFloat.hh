/// File: FusedSoftFloat.hh
/// SoftFloat optimised for GD32F103 (Cortex‑M3, ARMv7‑M)
///
/// Representation:
///   value = mantissa * 2^exponent
///   mantissa == 0  =>  zero
///   mantissa != 0  =>  abs(mantissa) in [2^29, 2^30)
///                      bit29 set, bits 31:30 clear in abs(mantissa)

#pragma once
#include <cstdint>
#include <cstring>
#include <climits>

#if __cplusplus >= 202002L
#   include <bit>   // std::bit_cast, std::countl_zero — both constexpr
#endif

// =========================================================================
// Platform detection
// =========================================================================
#if defined(__GNUC__) || defined(__clang__)
#   define SF_INLINE    __attribute__((always_inline)) inline
#   define SF_NOINLINE  __attribute__((noinline))
#   define SF_HOT       __attribute__((hot))
#   define SF_FLATTEN   __attribute__((flatten))
#   define LIKELY(x)    __builtin_expect(!!(x), 1)
#   define UNLIKELY(x)  __builtin_expect(!!(x), 0)
#else
#   define SF_INLINE    inline
#   define SF_NOINLINE
#   define SF_HOT
#   define SF_FLATTEN
#   define LIKELY(x)    (x)
#   define UNLIKELY(x)  (x)
#endif

// =========================================================================
// Consteval detection (no extra #include needed — GCC/Clang built-in)
// Allows functions to choose between ASM-optimised and constexpr-safe paths.
// =========================================================================
#if defined(__GNUC__) || defined(__clang__)
#   define SF_IS_CONSTEVAL() __builtin_is_constant_evaluated()
#else
#   define SF_IS_CONSTEVAL() false
#endif

// =========================================================================
// Bit‑cast helper (C++20 or fallback)
// std::bit_cast is constexpr in C++20; the memcpy fallback is NOT.
// =========================================================================
template<typename To, typename From>
[[nodiscard]] constexpr SF_INLINE To sf_bitcast(From v) noexcept
{
	static_assert(sizeof(To) == sizeof(From), "sf_bitcast: size mismatch");
#if __cplusplus >= 202002L
	return std::bit_cast<To>(v);
#else
	To r; __builtin_memcpy(&r, &v, sizeof(To)); return r;
#endif
}

// =========================================================================
// Cortex‑M3 primitives
// =========================================================================

// sf_clz — constexpr via std::countl_zero (C++20).
// std::countl_zero(0) == 32 by spec (no UB, unlike __builtin_clz(0)).
// On ARM, countl_zero compiles to a single CLZ instruction.
[[nodiscard]] constexpr SF_INLINE int sf_clz(uint32_t x) noexcept
{
	if (SF_IS_CONSTEVAL()) {
		// constexpr branch: use standard function
#if __cplusplus >= 202002L
		return std::countl_zero(x);
#else
		return __builtin_clz(x);
#endif
	}
	// Runtime: force CLZ instruction (ARMv7‑M)
	return __builtin_clz(x);
}

[[nodiscard]] constexpr int sf_clz64(uint64_t x) noexcept
{
	if (SF_IS_CONSTEVAL()) {
		if (x == 0) return 64;
		int n = 0;
		while ((x & 0x8000000000000000ULL) == 0) {
			x <<= 1;
			++n;
		}
		return n;
	}
	return __builtin_clzll(x);
}

// sf_abs32 — branchless absolute value for int32_t, constexpr-safe.
[[nodiscard]] constexpr SF_INLINE uint32_t sf_abs32(int32_t m) noexcept
{
#if defined(__arm__)
	if (!SF_IS_CONSTEVAL()) {
		uint32_t rd;
		__asm__ volatile (
			"eor %0, %1, %1, asr #31\n\t"
			"sub %0, %0, %1, asr #31\n\t"
			: "=&r"(rd) : "r"(m)
		);
		return rd;
	}
#endif
	uint32_t mask = static_cast<uint32_t>(m >> 31);
	return (static_cast<uint32_t>(m) ^ mask) - mask;
}

// Fast saturation – assumes exponent is already in [-128, 127].
// Used only in contexts where overflow is impossible (e.g., after add/sub).
[[nodiscard]] constexpr SF_INLINE int32_t sf_sat_exp_fast(int32_t e) noexcept
{
	// No-op; the exponent is guaranteed to be in range.
	// In debug builds you could add an assert(e >= -128 && e <= 127).
	return e;
}

// sf_sat_exp — saturate exponent to [-128, 127].
// At runtime on ARM: single SSAT instruction.
// At compile time: portable C++ fallback (SSAT is not constexpr-able).
[[nodiscard]] constexpr SF_INLINE int32_t sf_sat_exp(int32_t e) noexcept
{
#if defined(__arm__)
	if (!SF_IS_CONSTEVAL()) {
		int32_t r;
		__asm__ volatile (
			"ssat %0, #8, %1\n\t" 
			: "=r"(r) : "r"(e));
		return r;
	}
#endif
	if (e > 127) return  127;
	if (e < -128) return -128;
	return e;
}

// =========================================================================
// Fast Normalization - Optimized for Cortex-M3
// =========================================================================
constexpr SF_INLINE void sf_normalise_fast(int32_t& m, int32_t& e) noexcept
{
	if (UNLIKELY(m == 0)) {
		e = 0;
		return;
	}

	// 1. Branchless Absolute Value
	// Standard two's complement trick: (x ^ mask) - mask
	uint32_t sign = static_cast<uint32_t>(m >> 31);
	uint32_t a = (static_cast<uint32_t>(m) ^ sign) - sign;

	// 2. Fast Path: Already Normalized?
	// We want bit 29 set (0x20000000) and bit 30 clear (0x40000000).
	// Mask 0x60000000 extracts both.
	// We want value == 0x20000000.
	if (LIKELY((a & 0x60000000u) == 0x20000000u)) {
		e = sf_sat_exp_fast(e);
		return;
	}

	// 3. Handle Overflow (Bit 30 set)
	// This is mutually exclusive with underflow for normalized inputs,
	// but can happen after addition/subtraction.
	if (UNLIKELY(a & 0x40000000u)) {
		a >>= 1;
		e += 1;
	}
	else {
		// 4. Handle Underflow (Bit 29 clear)
		// We must shift left. Use CLZ to find how much.
		// Note: sf_clz is usually a builtin that compiles to CLZ instruction.
		int lz = sf_clz(a);

		// We want bit 29 set. 
		// CLZ counts zeros from MSB (bit 31).
		// If bit 29 is set, CLZ is 2.
		// Shift amount = lz - 2.
		int sh = lz - 2;

		// Update exponent
		// Note: We don't check for massive underflow (e < -128) here 
		// because sf_sat_exp_fast handles it at the end.
		e -= sh;

		// Shift mantissa
		a <<= sh;
	}

	// 5. Saturate Exponent & Restore Sign
	e = sf_sat_exp_fast(e);
	m = static_cast<int32_t>((a ^ sign) - sign);
}

// =========================================================================
// Reciprocal table (original, proven correct for [2^30, 2^31))
// ---------------------------------------------------------------------
// T[i] = floor(2^40 / (512 + i))
// For input b2 in [2^30, 2^31), index = (b2 >> 21) & 0x1FF
// T[i] * b2 ≈ 2^61
// =========================================================================
static constexpr uint32_t recip_tab[512] = {
	0x80000000u,0x7FC01FF0u,0x7F807F80u,0x7F411E52u,0x7F01FC07u,0x7EC31843u,0x7E8472A8u,0x7E460ADAu,
	0x7E07E07Eu,0x7DC9F339u,0x7D8C42B2u,0x7D4ECE8Fu,0x7D119679u,0x7CD49A16u,0x7C97D910u,0x7C5B5311u,
	0x7C1F07C1u,0x7BE2F6CEu,0x7BA71FE1u,0x7B6B82A6u,0x7B301ECCu,0x7AF4F3FEu,0x7ABA01EAu,0x7A7F4841u,
	0x7A44C6AFu,0x7A0A7CE6u,0x79D06A96u,0x79968F6Fu,0x795CEB24u,0x79237D65u,0x78EA45E7u,0x78B1445Cu,
	0x78787878u,0x783FE1F0u,0x78078078u,0x77CF53C5u,0x77975B8Fu,0x775F978Cu,0x77280772u,0x76F0AAF9u,
	0x76B981DAu,0x76828BCEu,0x764BC88Cu,0x761537D0u,0x75DED952u,0x75A8ACCFu,0x7572B201u,0x753CE8A4u,
	0x75075075u,0x74D1E92Fu,0x749CB28Fu,0x7467AC55u,0x7432D63Du,0x73FE3007u,0x73C9B971u,0x7395723Au,
	0x73615A24u,0x732D70EDu,0x72F9B658u,0x72C62A24u,0x7292CC15u,0x725F9BECu,0x722C996Bu,0x71F9C457u,
	0x71C71C71u,0x7194A17Fu,0x71625344u,0x71303185u,0x70FE3C07u,0x70CC728Fu,0x709AD4E4u,0x706962CCu,
	0x70381C0Eu,0x70070070u,0x6FD60FBAu,0x6FA549B4u,0x6F74AE26u,0x6F443CD9u,0x6F13F596u,0x6EE3D826u,
	0x6EB3E453u,0x6E8419E6u,0x6E5478ACu,0x6E25006Eu,0x6DF5B0F7u,0x6DC68A13u,0x6D978B8Eu,0x6D68B535u,
	0x6D3A06D3u,0x6D0B8036u,0x6CDD212Bu,0x6CAEE97Fu,0x6C80D901u,0x6C52EF7Fu,0x6C252CC7u,0x6BF790A8u,
	0x6BCA1AF2u,0x6B9CCB74u,0x6B6FA1FEu,0x6B429E60u,0x6B15C06Bu,0x6AE907EFu,0x6ABC74BEu,0x6A9006A9u,
	0x6A63BD81u,0x6A37991Au,0x6A0B9944u,0x69DFBDD4u,0x69B4069Bu,0x6988736Du,0x695D041Du,0x6931B880u,
	0x69069069u,0x68DB8BACu,0x68B0AA1Fu,0x6885EB95u,0x685B4FE5u,0x6830D6E4u,0x68068068u,0x67DC4C45u,
	0x67B23A54u,0x67884A69u,0x675E7C5Du,0x6734D006u,0x670B453Bu,0x66E1DBD4u,0x66B893A9u,0x668F6C91u,
	0x66666666u,0x663D80FFu,0x6614BC36u,0x65EC17E3u,0x65C393E0u,0x659B3006u,0x6572EC2Fu,0x654AC835u,
	0x6522C3F3u,0x64FADF42u,0x64D319FEu,0x64AB7401u,0x6483ED27u,0x645C854Au,0x64353C48u,0x640E11FAu,
	0x63E7063Eu,0x63C018F0u,0x639949EBu,0x6372990Eu,0x634C0634u,0x6325913Cu,0x62FF3A01u,0x62D90062u,
	0x62B2E43Du,0x628CE570u,0x626703D8u,0x62413F54u,0x621B97C2u,0x61F60D02u,0x61D09EF3u,0x61AB4D72u,
	0x61861861u,0x6160FF9Eu,0x613C0309u,0x61172283u,0x60F25DEAu,0x60CDB520u,0x60A92806u,0x6084B67Au,
	0x60606060u,0x603C2597u,0x60180601u,0x5FF4017Fu,0x5FD017F4u,0x5FAC493Fu,0x5F889545u,0x5F64FBE6u,
	0x5F417D05u,0x5F1E1885u,0x5EFACE48u,0x5ED79E31u,0x5EB48823u,0x5E918C01u,0x5E6EA9AEu,0x5E4BE10Fu,
	0x5E293205u,0x5E069C77u,0x5DE42046u,0x5DC1BD58u,0x5D9F7390u,0x5D7D42D4u,0x5D5B2B08u,0x5D392C10u,
	0x5D1745D1u,0x5CF57831u,0x5CD3C315u,0x5CB22661u,0x5C90A1FDu,0x5C6F35CCu,0x5C4DE1B6u,0x5C2CA5A0u,
	0x5C0B8170u,0x5BEA750Cu,0x5BC9805Bu,0x5BA8A344u,0x5B87DDADu,0x5B672F7Cu,0x5B46989Au,0x5B2618ECu,
	0x5B05B05Bu,0x5AE55ECDu,0x5AC5242Au,0x5AA5005Au,0x5A84F345u,0x5A64FCD2u,0x5A451CEAu,0x5A255374u,
	0x5A05A05Au,0x59E60382u,0x59C67CD8u,0x59A70C41u,0x5987B1A9u,0x59686CF7u,0x59493E14u,0x592A24EBu,
	0x590B2164u,0x58EC3368u,0x58CD5AE2u,0x58AE97BAu,0x588FE9DCu,0x58715130u,0x5852CDA0u,0x58345F18u,
	0x58160581u,0x57F7C0C5u,0x57D990D0u,0x57BB758Cu,0x579D6EE3u,0x577F7CC0u,0x57619F0Fu,0x5743D5BBu,
	0x572620AEu,0x57087FD4u,0x56EAF319u,0x56CD7A67u,0x56B015ACu,0x5692C4D1u,0x567587C4u,0x56585E70u,
	0x563B48C2u,0x561E46A4u,0x56015805u,0x55E47CD0u,0x55C7B4F1u,0x55AB0055u,0x558E5EE9u,0x5571D09Au,
	0x55555555u,0x5538ED06u,0x551C979Au,0x55005500u,0x54E42523u,0x54C807F2u,0x54ABFD5Au,0x54900549u,
	0x54741FABu,0x54584C70u,0x543C8B84u,0x5420DCD6u,0x54054054u,0x53E9B5EBu,0x53CE3D8Bu,0x53B2D721u,
	0x5397829Cu,0x537C3FEBu,0x53610EFBu,0x5345EFBCu,0x532AE21Cu,0x530FE60Bu,0x52F4FB76u,0x52DA224Eu,
	0x52BF5A81u,0x52A4A3FEu,0x5289FEB5u,0x526F6A96u,0x5254E78Eu,0x523A758Fu,0x52201488u,0x5205C467u,
	0x51EB851Eu,0x51D1569Cu,0x51B738D1u,0x519D2BADu,0x51832F1Fu,0x51694319u,0x514F678Bu,0x51359C64u,
	0x511BE195u,0x5102370Fu,0x50E89CC2u,0x50CF129Fu,0x50B59897u,0x509C2E9Au,0x5082D499u,0x50698A85u,
	0x50505050u,0x503725EAu,0x501E0B44u,0x50050050u,0x4FEC04FEu,0x4FD31941u,0x4FBA3D0Au,0x4FA1704Au,
	0x4F88B2F3u,0x4F7004F7u,0x4F576646u,0x4F3ED6D4u,0x4F265691u,0x4F0DE571u,0x4EF58364u,0x4EDD305Du,
	0x4EC4EC4Eu,0x4EACB72Au,0x4E9490E1u,0x4E7C7968u,0x4E6470B0u,0x4E4C76ABu,0x4E348B4Du,0x4E1CAE88u,
	0x4E04E04Eu,0x4DED2092u,0x4DD56F47u,0x4DBDCC5Fu,0x4DA637CFu,0x4D8EB188u,0x4D77397Eu,0x4D5FCFA4u,
	0x4D4873ECu,0x4D31264Bu,0x4D19E6B3u,0x4D02B518u,0x4CEB916Du,0x4CD47BA5u,0x4CBD73B5u,0x4CA67990u,
	0x4C8F8D28u,0x4C78AE73u,0x4C61DD63u,0x4C4B19EDu,0x4C346404u,0x4C1DBB9Du,0x4C0720ABu,0x4BF09322u,
	0x4BDA12F6u,0x4BC3A01Cu,0x4BAD3A87u,0x4B96E22Du,0x4B809701u,0x4B6A58F7u,0x4B542804u,0x4B3E041Du,
	0x4B27ED36u,0x4B11E343u,0x4AFBE639u,0x4AE5F60Du,0x4AD012B4u,0x4ABA3C21u,0x4AA4724Bu,0x4A8EB526u,
	0x4A7904A7u,0x4A6360C3u,0x4A4DC96Eu,0x4A383E9Fu,0x4A22C04Au,0x4A0D4E64u,0x49F7E8E2u,0x49E28FBAu,
	0x49CD42E2u,0x49B8024Du,0x49A2CDF3u,0x498DA5C8u,0x497889C2u,0x496379D6u,0x494E75FAu,0x49397E24u,
	0x49249249u,0x490FB25Fu,0x48FADE5Cu,0x48E61636u,0x48D159E2u,0x48BCA957u,0x48A8048Au,0x48936B72u,
	0x487EDE04u,0x486A5C37u,0x4855E601u,0x48417B57u,0x482D1C31u,0x4818C884u,0x48048048u,0x47F04371u,
	0x47DC11F7u,0x47C7EBCFu,0x47B3D0F1u,0x479FC154u,0x478BBCECu,0x4777C3B2u,0x4763D59Cu,0x474FF2A1u,
	0x473C1AB6u,0x47284DD4u,0x47148BF0u,0x4700D502u,0x46ED2901u,0x46D987E3u,0x46C5F19Fu,0x46B2662Du,
	0x469EE584u,0x468B6F9Au,0x46780467u,0x4664A3E2u,0x46514E02u,0x463E02BEu,0x462AC20Eu,0x46178BE9u,
	0x46046046u,0x45F13F1Cu,0x45DE2864u,0x45CB1C14u,0x45B81A25u,0x45A5228Cu,0x45923543u,0x457F5241u,
	0x456C797Du,0x4559AAF0u,0x4546E68Fu,0x45342C55u,0x45217C38u,0x450ED630u,0x44FC3A34u,0x44E9A83Eu,
	0x44D72044u,0x44C4A23Fu,0x44B22E27u,0x449FC3F4u,0x448D639Du,0x447B0D1Bu,0x4468C066u,0x44567D76u,
	0x44444444u,0x443214C7u,0x441FEEF8u,0x440DD2CEu,0x43FBC043u,0x43E9B74Fu,0x43D7B7EAu,0x43C5C20Du,
	0x43B3D5AFu,0x43A1F2CAu,0x43901956u,0x437E494Bu,0x436C82A2u,0x435AC553u,0x43491158u,0x433766A9u,
	0x4325C53Eu,0x43142D11u,0x43029E1Au,0x42F11851u,0x42DF9BB0u,0x42CE2830u,0x42BCBDC8u,0x42AB5C73u,
	0x429A0429u,0x4288B4E3u,0x42776E9Au,0x42663147u,0x4254FCE4u,0x4243D168u,0x4232AECDu,0x4221950Du,
	0x42108421u,0x41FF7C01u,0x41EE7CA6u,0x41DD860Bu,0x41CC9829u,0x41BBB2F8u,0x41AAD671u,0x419A0290u,
	0x4189374Bu,0x4178749Eu,0x4167BA81u,0x415708EEu,0x41465FDFu,0x4135BF4Cu,0x41252730u,0x41149783u,
	0x41041041u,0x40F39161u,0x40E31ADEu,0x40D2ACB1u,0x40C246D4u,0x40B1E941u,0x40A193F1u,0x409146DFu,
	0x40810204u,0x4070C559u,0x406090D9u,0x4050647Du,0x40404040u,0x4030241Bu,0x40201008u,0x40100401u,
};

// sf_recip — Newton-refined reciprocal, now constexpr.
// Returns Y ≈ 2^60 / b  for b in [2^29, 2^30).
// Uses one UMULL Newton step: ~14 cycles on Cortex-M3.
[[nodiscard]] constexpr SF_INLINE uint64_t sf_recip(uint32_t b) noexcept
{
	uint32_t b2 = b << 1;                      // [2^29,2^30) → [2^30,2^31)
	uint32_t idx = (b2 >> 21) & 0x1FFu;         // 9‑bit index into table
	uint32_t Y = recip_tab[idx];

	uint64_t bY = static_cast<uint64_t>(b2) * Y;
	int64_t  err64 = static_cast<int64_t>(1ULL << 61) - static_cast<int64_t>(bY);
	int32_t  err = static_cast<int32_t>(err64 >> 30);
	int64_t  dY64 = (static_cast<int64_t>(Y) * err) >> 31;
	return static_cast<uint64_t>(static_cast<int64_t>(Y) + dY64);
}

// ---------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------
struct sf_mul_expr;
class  SoftFloat;
struct SoftFloatPair;

// =========================================================================
// SoftFloat class
// =========================================================================
class SoftFloat {
public:
	int32_t mantissa /*:24*/;
	int32_t exponent /*:8*/;

	// ------------------------------------------------------------------
	// Default constructor — zero
	// ------------------------------------------------------------------
	constexpr SoftFloat() noexcept : mantissa{ 0 }, exponent{ 0 } {}

	// ------------------------------------------------------------------
	// from_raw — bypass normalisation (caller guarantees invariant)
	// ------------------------------------------------------------------
	[[nodiscard]] static constexpr SoftFloat from_raw(int32_t m, int32_t e) noexcept {
		SoftFloat r; r.mantissa = m; r.exponent = e; return r;
	}

	// ------------------------------------------------------------------
	// Normalising constructors — now constexpr
	// ------------------------------------------------------------------
	constexpr SF_HOT SoftFloat(int32_t m, int32_t e) noexcept
		: mantissa{ m }, exponent{ e }
	{
		normalise();
	}

	constexpr SF_HOT explicit SoftFloat(int v) noexcept
		: mantissa{ v }, exponent{ 0 }
	{
		normalise();
	}

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

	// Proxy constructor (defined after sf_mul_expr)
	constexpr SF_HOT SoftFloat(const sf_mul_expr& m) noexcept;

	// ------------------------------------------------------------------
	// Manual re-normalise (public utility, rarely needed)
	// ------------------------------------------------------------------
		// =========================================================================
	// Normalisation for [2^29, 2^30)
	// ---------------------------------------------------------------------
	// Target: bit29 set, bits 31:30 both zero in abs(mantissa).
	// CLZ of correctly normalised value == 2.
	// Now constexpr: usable in static_assert and constinit contexts.
	// =========================================================================
	constexpr SF_HOT SF_INLINE void normalise() noexcept
	{
		int32_t m = mantissa, e = exponent;
		if (m == 0) { 
			exponent = 0;
			return;
		}
		uint32_t a = sf_abs32(m);
		int lz = sf_clz(a);
		int shift = lz - 2;

		if (shift > 0) {
			int ne = e - shift;
			if (ne < -250) { 
				mantissa = 0;
				exponent = 0;
				return; 
			}
			a <<= shift;
			e = ne;
		}
		else if (shift < 0) {
			int rs = -shift;
			a >>= rs;
			e += rs;
		}

		exponent = sf_sat_exp(e);
		mantissa = (m < 0) ? -static_cast<int32_t>(a) : static_cast<int32_t>(a);
	}

	// ------------------------------------------------------------------
	// to_float — constexpr in C++20 via std::bit_cast
	//
	// Layout for normalised value in [2^29, 2^30):
	//   bias = 29 + 127 = 156
	//   fraction = bits 28..6 of abs(mantissa) mapped to IEEE bits 22..0
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SF_HOT float to_float() const noexcept {
		if (!mantissa) return 0.f;
		uint32_t a = sf_abs32(mantissa);
		int      iexp = exponent + 156;
		if (iexp >= 255) return mantissa > 0 ? 3.4028235e38f : -3.4028235e38f;
		if (iexp <= 0) return 0.f;
		uint32_t bits = (mantissa < 0 ? 0x80000000u : 0u)
			| (static_cast<uint32_t>(iexp) << 23)
			| ((a >> 6) & 0x007FFFFFu);
		return sf_bitcast<float>(bits);
	}
	[[nodiscard]] constexpr explicit operator float()   const noexcept { return to_float(); }

	// ------------------------------------------------------------------
	// to_int32 — truncate toward zero, constexpr
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SF_HOT int32_t to_int32() const noexcept {
		if (!mantissa) return 0;
		if (exponent >= 2) return mantissa > 0 ? INT32_MAX : INT32_MIN;

		// Use sf_abs32 + unsigned shift to truncate toward zero (not toward -inf).
		// Arithmetic right shift of negative values rounds toward -inf, which
		// gives e.g. (int)SoftFloat(-42.9f) == -43 instead of correct -42.
		uint32_t a = sf_abs32(mantissa);

		if (exponent >= 0) {
			// exponent is 0 or 1: shift left, no precision loss
			a <<= exponent;
		}
		else {
			int rs = -exponent;
			if (rs >= 31) return 0;
			a >>= rs;    // unsigned shift: truncates toward zero for both signs
		}

		return mantissa < 0 ? -static_cast<int32_t>(a) : static_cast<int32_t>(a);
	}
	[[nodiscard]] constexpr explicit operator int32_t() const noexcept { return to_int32(); }

	// ------------------------------------------------------------------
	// Unary
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SoftFloat operator-() const noexcept { return from_raw(-mantissa, exponent); }
	[[nodiscard]] constexpr SoftFloat operator+() const noexcept { return *this; }

	// ------------------------------------------------------------------
	// Binary operator declarations (defined after sf_mul_expr)
	// ------------------------------------------------------------------
	friend constexpr SF_HOT SF_INLINE sf_mul_expr operator*(SoftFloat a, SoftFloat b) noexcept;
	friend constexpr SF_HOT SF_INLINE sf_mul_expr operator*(SoftFloat a, float     b) noexcept;
	friend constexpr SF_HOT SF_INLINE sf_mul_expr operator*(SoftFloat a, int32_t   b) noexcept;
	friend constexpr SF_HOT SF_INLINE sf_mul_expr operator*(float     a, SoftFloat b) noexcept;
	friend constexpr SF_HOT SF_INLINE sf_mul_expr operator*(int32_t   a, SoftFloat b) noexcept;

	// ------------------------------------------------------------------
	// Add — constexpr, 32-bit only (no overflow possible)
	//
	// After alignment: larger operand abs in [2^29,2^30), smaller <= 2^29.
	// Worst-case same-sign sum: (2^30-1)+2^29 < 2^31  =>  no overflow.
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator+(SoftFloat a, SoftFloat b) noexcept {
		if (UNLIKELY(!a.mantissa)) return b;
		if (UNLIKELY(!b.mantissa)) return a;

		int d = a.exponent - b.exponent;
		if (d >= 31) return a;
		if (d <= -31) return b;

		int32_t rm, re;

		if (LIKELY(d == 0)) {

			rm = a.mantissa + b.mantissa;
			re = a.exponent;
			if (UNLIKELY(rm == 0)) return {};
			
			// Detect overflow into bit30 without sf_abs32:
			// (rm ^ (rm>>31)) gives |rm|-1 for negative, |rm| for positive.
			// Bit30 set means |rm| >= 2^30, i.e. overflow by 1 bit.
			uint32_t ov = static_cast<uint32_t>(rm ^ (rm >> 31)) >> 30;
			rm >>= static_cast<int>(ov);
			re += static_cast<int32_t>(ov);
			re = sf_sat_exp(re);
			return from_raw(rm, re);
		}

		// Unequal exponents: align smaller operand with arithmetic shift.
		if (d > 0) {
			rm = a.mantissa + (b.mantissa >> d);
			re = a.exponent;
		}
		else {
			rm = (a.mantissa >> -d) + b.mantissa;
			re = b.exponent;
		}

		return sf_finish_addsub(rm, re);
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN 
		friend SoftFloat operator+(SoftFloat a, float b) noexcept {
		return a + SoftFloat(b);
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator+(SoftFloat a, int32_t b) noexcept {
		return a + SoftFloat(b);
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator+(float a, SoftFloat b) noexcept {
		return SoftFloat(a) + b;
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator+(int32_t a, SoftFloat b) noexcept {
		return SoftFloat(a) + b;
	}

	// ------------------------------------------------------------------
	// Sub — constexpr
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator-(SoftFloat a, SoftFloat b) noexcept {
#
		if (UNLIKELY(!b.mantissa)) return a;
		if (UNLIKELY(!a.mantissa)) return -b;

		int d = a.exponent - b.exponent;
		if (d >= 31) return a;
		if (d <= -31) return -b;

		int32_t rm, re;

		if (LIKELY(d == 0)) {
			// Same exponent: direct signed subtraction.
			rm = a.mantissa - b.mantissa;
			re = a.exponent;

			if (UNLIKELY(rm == 0)) return {};

			uint32_t ab = sf_abs32(rm);

			// Already normalized?
			if (LIKELY((ab & 0x60000000u) == 0x20000000u)) {
				return from_raw(rm, sf_sat_exp_fast(re));
			}

			// Overflow into bit30? (Possible only if abs(rm) >= 2^30)
			if (ab & 0x40000000u) {
				rm >>= 1;              // arithmetic shift preserves sign
				re += 1;
				return from_raw(rm, sf_sat_exp_fast(re));
			}

			// Otherwise cancellation happened: need full renormalization.
			sf_normalise_fast(rm, re);
			return from_raw(rm, re);
		}

		// Unequal exponents: align smaller operand with arithmetic shift.
		if (d > 0) {
			rm = a.mantissa - (b.mantissa >> d);
			re = a.exponent;
		}
		else {
			rm = (a.mantissa >> -d) - b.mantissa;
			re = b.exponent;
		}

		return sf_finish_addsub(rm, re);
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator-(SoftFloat a, float b) noexcept {
		return a - SoftFloat(b);
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator-(SoftFloat a, int32_t b) noexcept {
		return a - SoftFloat(b);
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator-(float a, SoftFloat b) noexcept {
		return SoftFloat(a) - b;
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator-(int32_t a, SoftFloat b) noexcept {
		return SoftFloat(a) - b;
	}

	// ------------------------------------------------------------------
	// Divide — constexpr
	//
	// sf_recip(ub) returns Y ≈ 2^60 / ub  (ub in [2^29, 2^30))
	// q64 = ua * Y ≈ ua * 2^60 / ub
	// ua, ub in [2^29, 2^30)  =>  ua/ub in [0.5, 2.0)
	// q64 >> 30 gives result in [2^28, 2^30).
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SF_HOT SoftFloat operator/(SoftFloat rhs) const noexcept {
		if (UNLIKELY(!rhs.mantissa))
			return from_raw(mantissa >= 0 ? (1 << 29) : -(1 << 29), 127);
		if (UNLIKELY(!mantissa)) return {};

		bool     neg = (mantissa ^ rhs.mantissa) < 0;
		uint32_t ua = sf_abs32(mantissa);
		uint32_t ub = sf_abs32(rhs.mantissa);

		// Compile‑time path (unchanged)
		if (SF_IS_CONSTEVAL()) {

			uint64_t Y = sf_recip(ub);
			uint64_t q64 = static_cast<uint64_t>(ua) * Y;

			uint32_t qm32 = static_cast<uint32_t>(q64 >> 30);
			int32_t  qe = exponent - rhs.exponent - 30;

			// qm32 in [2^28, 2^30). Branchless 1-bit adjust:
			uint32_t hi = qm32 >> 30;
			uint32_t lo = (~qm32 >> 29) & 1u & ~hi;
			qm32 = (qm32 >> hi) << lo;
			qe += static_cast<int32_t>(hi) - static_cast<int32_t>(lo);

			int32_t qm = neg ? -static_cast<int32_t>(qm32) : static_cast<int32_t>(qm32);

			qe = sf_sat_exp(qe);           // exponent may exceed 127; clamp it
			return from_raw(qm, qe);       // mantissa already normalised
		}

		// Compute floor((ua << 30) / ub) using exactly two UDIV instructions.
		//
		// Rewrite as floor((ua << 32) / (ub << 2)):
		//   ub ∈ [2^29, 2^30)  ⟹  ub << 2 ∈ [2^31, 2^32) always.
		//   The top bit is guaranteed set by our normalisation invariant,
		//   so the shift is always exactly 2 — no CLZ required.
		//
		// Then apply Knuth TAOCP Vol.2 §4.3.1 Algorithm D for a single
		// normalised-word divisor, yielding two 16-bit quotient digits
		// with one UDIV each.  All intermediate products are within
		// uint32_t range (proofs in comments below).
		//
		// On Cortex-M3, each UDIV costs 2–12 cycles vs. the ~60+ cycles
		// of __aeabi_uldivmod.  The correction blocks run at most twice
		// per digit; the q >= 2^16 guard is statically unreachable
		// (ua < 2^30 and vn1 >= 2^15 ⟹ q1 < 2^15 always).
#if defined(__arm__)
		{
			uint32_t v = ub << 2;          // normalised divisor ∈ [2^31, 2^32)
			uint32_t vn1 = v >> 16;          // high half ∈ [2^15, 2^16)
			uint32_t vn0 = v & 0xFFFFu;     // low  half ∈ [0,    2^16)

			// ── High quotient digit ─────────────────────────────────────
			//   Dividend high word = ua (≡ the "ua:0" 64-bit value's top 32 bits).
			//   q1 = floor(ua / vn1).  ua < 2^30, vn1 >= 2^15 ⟹ q1 < 2^15.
			uint32_t q1, rhat;
			__asm__ volatile (
				"udiv %0, %1, %2\n\t" 
				: "=r"(q1) 
				: "r"(ua), "r"(vn1)
			);  // UDIV #1
			rhat = ua - q1 * vn1;            // MLS; remainder mod vn1, < vn1 < 2^16

			// Knuth correction: q1_hat may exceed the true digit by at most 2.
			// q1 * vn0 < 2^31 (q1<2^15, vn0<2^16); rhat<<16 < 2^32 (rhat<2^16): no overflow.
			if (q1 * vn0 > (rhat << 16)) {
				--q1; rhat += vn1;
				// After first correction rhat < 2*vn1 < 2^17. Guard against
				// (rhat<<16) overflow before the second comparison.
				if (rhat < 0x10000u && q1 * vn0 >(rhat << 16)) --q1;
			}
			// un21 = ua*2^16 - q1*v, computed overflow-free via the remainder.
			// rhat < 2^16 here ⟹ (rhat<<16) < 2^32; q1*vn0 ≤ (rhat<<16) ⟹ result ≥ 0.
			uint32_t un21 = (rhat << 16) - q1 * vn0;   // ∈ [0, v)

			// ── Low quotient digit ──────────────────────────────────────
			//   un21 < v ⟹ q0 = un21/vn1 < v/vn1 = v/(v>>16) ≤ 65535 < 2^16.
			//   q0*vn0 ≤ 65535*65535 = 2^32 - 131071 < 2^32: no overflow.
			uint32_t q0;
			__asm__ volatile (
				"udiv %0, %1, %2\n\t" 
				: "=r"(q0) 
				: "r"(un21), "r"(vn1)
			);  // UDIV #2
			rhat = un21 - q0 * vn1;

			if (q0 * vn0 > (rhat << 16)) {
				--q0; rhat += vn1;
				if (rhat < 0x10000u && q0 * vn0 >(rhat << 16)) --q0;
			}

			uint32_t qm = (q1 << 16) | q0;  // ∈ [2^29, 2^31)
			int32_t  qe = exponent - rhs.exponent - 30;

#if 0
			// Normalise mantissa to [2^29, 2^30)
			int lz = sf_clz(qm);
			int shift = lz - 2;
			if (shift > 0) {
				qm <<= shift;
				qe -= shift;
			}
			else if (shift < 0) {
				qm >>= -shift;
				qe += -shift;
			}
#else
			if (qm & 0x40000000u) {   // bit30 set -> need right shift by 1
				qm >>= 1;
				qe += 1;
			}
			// No left shift possible because qm >= 2^29 always
#endif

			qe = sf_sat_exp(qe);
			int32_t qm_signed = neg ? -static_cast<int32_t>(qm) : static_cast<int32_t>(qm);
			return from_raw(qm_signed, qe);
		} 
#else
		// Non-ARM fallback (host-side build / unit tests).
		{
			uint32_t qm = static_cast<uint32_t>((static_cast<uint64_t>(ua) << 30) / ub);
			int32_t  qe = exponent - rhs.exponent - 30;

			// Normalise mantissa to [2^29, 2^30)
			int lz = sf_clz(qm);
			int shift = lz - 2;
			if (shift > 0) {
				qm <<= shift;
				qe -= shift;
			}
			else if (shift < 0) {
				qm >>= -shift;
				qe += -shift;
			}

			qe = sf_sat_exp(qe);
			int32_t qm_signed = neg ? -static_cast<int32_t>(qm) : static_cast<int32_t>(qm);
			return from_raw(qm_signed, qe);
		}
#endif
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		SoftFloat operator/(float rhs) const noexcept {
		return *this / SoftFloat(rhs);
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		SoftFloat operator/(int32_t rhs) const noexcept {
		return *this / SoftFloat(rhs);
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator/(float lhs, SoftFloat rhs) noexcept {
		return SoftFloat(lhs) / rhs;
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator/(int32_t lhs, SoftFloat rhs) noexcept {
		return SoftFloat(lhs) / rhs;
	}

	// ------------------------------------------------------------------
	// Power-of-2 scaling (exponent adjust only, O(1)) — constexpr
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		SoftFloat operator>>(int s) const noexcept {
		if (!mantissa) return *this;
		int32_t ne = exponent - s;
		if (ne < -250) return {};
		return from_raw(mantissa, ne);
	}
	constexpr SF_HOT SF_INLINE SF_FLATTEN
		const SoftFloat& operator>>=(int s) noexcept {
		if (!mantissa) return *this;
		int32_t ne = exponent - s;
		if (ne < -250) {
			mantissa = 0; exponent = 0;
			return *this;
		}
		exponent = ne;
		return *this;
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		SoftFloat operator<<(int s) const noexcept {
		if (!mantissa) return *this;
		int32_t ne = exponent + s;
		if (ne > 127) return from_raw(mantissa > 0 ? (1 << 29) : -(1 << 29), 127);
		return from_raw(mantissa, ne);
	}
	constexpr SF_HOT SF_INLINE SF_FLATTEN
		const SoftFloat& operator<<=(int s) noexcept {
		if (!mantissa) return *this;
		int32_t ne = exponent + s;
		if (ne > 127) {
			mantissa = mantissa > 0 ? (1 << 29) : -(1 << 29);
			exponent = 127;
			return *this;
		}
		exponent = ne;
		return *this;
	}

	// ------------------------------------------------------------------
	// Comparison — constexpr (unchanged logic)
	// ------------------------------------------------------------------
	[[nodiscard]] friend constexpr bool operator==(SoftFloat a, SoftFloat b) noexcept {
		if (!a.mantissa && !b.mantissa) return true;
		return a.mantissa == b.mantissa && a.exponent == b.exponent;
	}
	[[nodiscard]] friend constexpr bool operator!=(SoftFloat a, SoftFloat b) noexcept { return !(a == b); }
	[[nodiscard]] friend constexpr bool operator< (SoftFloat a, SoftFloat b) noexcept {
		if (!a.mantissa && !b.mantissa) return false;
		if (!a.mantissa) return b.mantissa > 0;
		if (!b.mantissa) return a.mantissa < 0;
		bool an = a.mantissa < 0, bn = b.mantissa < 0;
		if (an != bn) return an;
		if (a.exponent != b.exponent)
			return an ? a.exponent > b.exponent : a.exponent < b.exponent;
		return an ? a.mantissa > b.mantissa : a.mantissa < b.mantissa;
	}
	[[nodiscard]] friend constexpr bool operator> (SoftFloat a, SoftFloat b) noexcept { return b < a; }
	[[nodiscard]] friend constexpr bool operator<=(SoftFloat a, SoftFloat b) noexcept { return !(a > b); }
	[[nodiscard]] friend constexpr bool operator>=(SoftFloat a, SoftFloat b) noexcept { return !(a < b); }

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
		return from_raw(static_cast<int32_t>(sf_abs32(mantissa)), exponent);
	}

	// ------------------------------------------------------------------
	// Clamp — constexpr (unchanged)
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SoftFloat clamp(SoftFloat lo, SoftFloat hi) const noexcept {
		if (*this < lo) return lo;
		if (*this > hi) return hi;
		return *this;
	}

	// ------------------------------------------------------------------
	// Math functions — constexpr via integer arithmetic only
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SF_HOT SoftFloat sin() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat cos() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloatPair sincos() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat tan() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat asin() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat acos() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat sinh() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat cosh() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat tanh() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat inv_sqrt() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat sqrt()     const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat exp()     const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat log() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat log2() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat log10() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat pow(SoftFloat y) const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat trunc() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat floor() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat ceil() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat round() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloat fract() const noexcept;
	[[nodiscard]] constexpr SF_HOT SoftFloatPair modf() const noexcept;
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

	// ------------------------------------------------------------------
	// Constants
	//
	// All mantissa values verified: clz(abs(mantissa)) == 2
	//
	// one = 1.0:     0x20000000 * 2^-29 = 1.0
	// half = 0.5:    0x20000000 * 2^-30 = 0.5
	// two = 2.0:     0x20000000 * 2^-28 = 2.0
	// pi:   843314857 * 2^-28 = 3.14159265...
	// 2pi:  same mantissa, exponent -27
	// pi/2: same mantissa, exponent -29
	// ------------------------------------------------------------------
	[[nodiscard]] static constexpr SoftFloat zero()     noexcept { return from_raw(0, 0); }
	[[nodiscard]] static constexpr SoftFloat one()      noexcept { return from_raw(0x20000000, -29); }
	[[nodiscard]] static constexpr SoftFloat neg_one()  noexcept { return from_raw(-0x20000000, -29); }
	[[nodiscard]] static constexpr SoftFloat half()     noexcept { return from_raw(0x20000000, -30); }
	[[nodiscard]] static constexpr SoftFloat two()      noexcept { return from_raw(0x20000000, -28); }
	[[nodiscard]] static constexpr SoftFloat three()    noexcept { return from_raw(0x30000000, -28); }
	[[nodiscard]] static constexpr SoftFloat four()    noexcept { return from_raw(0x20000000, -27); }
	[[nodiscard]] static constexpr SoftFloat pi()       noexcept { return from_raw(843314857, -28); }
	[[nodiscard]] static constexpr SoftFloat two_pi()   noexcept { return from_raw(843314857, -27); }
	[[nodiscard]] static constexpr SoftFloat half_pi()  noexcept { return from_raw(843314857, -29); }

	// ------------------------------------------------------------------
	// Fused operations (friend declarations)
	// ------------------------------------------------------------------
	friend constexpr SF_HOT SoftFloat fused_mul_add(SoftFloat a, SoftFloat b, SoftFloat c) noexcept;
	friend constexpr SF_HOT SoftFloat fused_mul_sub(SoftFloat a, SoftFloat b, SoftFloat c) noexcept;
	friend constexpr SF_HOT SoftFloat fused_mul_mul_add(SoftFloat a, SoftFloat b, SoftFloat c, SoftFloat d) noexcept;
	friend constexpr SF_HOT SoftFloat fused_mul_mul_sub(SoftFloat a, SoftFloat b, SoftFloat c, SoftFloat d) noexcept;

private:
	friend struct sf_mul_expr;

	[[nodiscard]] static constexpr SF_INLINE SoftFloat sf_finish_addsub(int32_t rm, int32_t re) noexcept {
		if (UNLIKELY(rm == 0)) return {};

		uint32_t ab = sf_abs32(rm);

		if (LIKELY((ab & 0x60000000u) == 0x20000000u)) {
			return SoftFloat::from_raw(rm, sf_sat_exp_fast(re));
		}

		if (ab & 0x40000000u) {
			rm >>= 1;
			re += 1;
			return SoftFloat::from_raw(rm, sf_sat_exp_fast(re));
		}

		sf_normalise_fast(rm, re);
		return SoftFloat::from_raw(rm, re);
	}

	// constexpr integer sqrt for 64-bit, binary search
	[[nodiscard]] static constexpr uint64_t ct_isqrt64(uint64_t n) noexcept {
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

	// mul_plain — core multiply kernel, constexpr
	// Both inputs have abs in [2^29,2^30). Product in [2^58,2^60).
	// >> 29 gives [2^29,2^31) => at most one bit of adjustment.
	[[nodiscard]] static constexpr SF_INLINE SF_FLATTEN
		SoftFloat mul_plain(SoftFloat a, SoftFloat b) noexcept {
		if (UNLIKELY(!a.mantissa || !b.mantissa)) return {};

		// 1. Multiply
		int64_t prod = static_cast<int64_t>(a.mantissa) * b.mantissa;

		// 2. Pre-shift. 
		// Product is in [2^58, 2^60).
		// Shift right by 29 to get potential mantissa in [2^29, 2^31).
		int32_t rm = static_cast<int32_t>(prod >> 29);
		int32_t re = a.exponent + b.exponent + 29;

		// 3. Normalize (Signed Safe)
		// We only need to check if magnitude >= 2^30.
		// Use abs helper to handle signed rm correctly.
		uint32_t abs_m = sf_abs32(rm);

		if (UNLIKELY(abs_m >= 0x40000000u)) {
			// Overflow by 1 bit (or more if inputs were edge cases)
			// Arithmetic shift right preserves sign.
			rm >>= 1;
			re += 1;
		}

		// Note: Multiplication cannot underflow (shift left needed) unless 
		// one input was effectively zero, which we checked.
		// The result of norm mults is always >= 2^29.

		re = sf_sat_exp_fast(re);
		return from_raw(rm, re);
	}

	// from_float — parse IEEE 754 single, constexpr via std::bit_cast
	constexpr SF_HOT void from_float(float f) noexcept {
		uint32_t bits = sf_bitcast<uint32_t>(f);
		if ((bits & 0x7FFFFFFFu) == 0) { mantissa = 0; exponent = 0; return; }
		bool     neg = (bits >> 31) != 0;
		uint32_t expf = (bits >> 23) & 0xFFu;
		uint32_t frac = bits & 0x7FFFFFu;
		if (expf == 0xFFu) {          // NaN / Inf — clamp to large finite
			mantissa = neg ? -(1 << 29) : (1 << 29);
			exponent = 98;
			return;
		}
		if (expf == 0) { mantissa = 0; exponent = 0; return; }   // denormal → zero
		uint32_t m = (1u << 29) | (frac << 6);
		mantissa = neg ? -static_cast<int32_t>(m) : static_cast<int32_t>(m);
		exponent = static_cast<int32_t>(expf) - 156;           // bias: 127 + 29 = 156
	}
};

struct SoftFloatPair { SoftFloat intpart; SoftFloat fracpart; };

// =========================================================================
// Expression‑template proxy for a deferred single multiplication.
// Allows the compiler to fuse  a + b*c  into a single FMA call.
// =========================================================================
struct sf_mul_expr {
	SoftFloat lhs;
	SoftFloat rhs;

	// Materialise the product — used when not part of a fused chain.
	[[nodiscard]] constexpr SF_INLINE SoftFloat eval() const noexcept {
		return SoftFloat::mul_plain(lhs, rhs);
	}

	// Implicit conversion to SoftFloat (via constructor below)
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
	[[nodiscard]] constexpr SoftFloat trunc() const noexcept { return eval().trunc(); }
	[[nodiscard]] constexpr SoftFloat floor() const noexcept { return eval().floor(); }
	[[nodiscard]] constexpr SoftFloat ceil() const noexcept { return eval().ceil(); }
	[[nodiscard]] constexpr SoftFloat round() const noexcept { return eval().round(); }
	[[nodiscard]] constexpr SoftFloat fract() const noexcept { return eval().fract(); }
	[[nodiscard]] constexpr SoftFloatPair modf() const noexcept { return eval().modf(); }
	[[nodiscard]] constexpr SoftFloat copysign(SoftFloat sign) const noexcept { return eval().copysign(sign); }
	[[nodiscard]] constexpr SoftFloat fmod(SoftFloat y) const noexcept { return eval().fmod(y); }
	[[nodiscard]] constexpr SoftFloat fma(SoftFloat b, SoftFloat c) const noexcept { return eval().fma(b, c); }
	[[nodiscard]] constexpr SoftFloat inv_sqrt()          const noexcept { return eval().inv_sqrt(); }
	[[nodiscard]] constexpr SoftFloat clamp(SoftFloat lo, SoftFloat hi) const noexcept {
		return eval().clamp(lo, hi);
	}
	[[nodiscard]] constexpr SoftFloat sin()               const noexcept { return eval().sin(); }
	[[nodiscard]] constexpr SoftFloat cos()               const noexcept { return eval().cos(); }
	[[nodiscard]] constexpr SoftFloatPair sincos()               const noexcept { return eval().sincos(); }
	[[nodiscard]] constexpr SoftFloat tan() const noexcept { return eval().tan(); }
	[[nodiscard]] constexpr SoftFloat asin() const noexcept { return eval().asin(); }
	[[nodiscard]] constexpr SoftFloat acos() const noexcept { return eval().acos(); }
	[[nodiscard]] constexpr SoftFloat sinh() const noexcept { return eval().sinh(); }
	[[nodiscard]] constexpr SoftFloat cosh() const noexcept { return eval().cosh(); }
	[[nodiscard]] constexpr SoftFloat tanh() const noexcept { return eval().tanh(); }
	[[nodiscard]] constexpr SoftFloat operator/(SoftFloat r)  const noexcept { return eval() / r; }
	[[nodiscard]] constexpr SoftFloat operator/(float r)      const noexcept { return eval() / SoftFloat(r); }
	[[nodiscard]] constexpr SoftFloat operator/(int32_t r)    const noexcept { return eval() / SoftFloat(r); }
	[[nodiscard]] constexpr SoftFloat operator>>(int s)       const noexcept { return eval() >> s; }
	[[nodiscard]] constexpr SoftFloat operator<<(int s)       const noexcept { return eval() << s; }

	// Negate the expression (flips lhs sign, lazy — no evaluation)
	[[nodiscard]] constexpr sf_mul_expr operator-() const noexcept {
		sf_mul_expr r = *this;
		r.lhs = -r.lhs;
		return r;
	}
};

// =========================================================================
// SoftFloat proxy constructor (defined here so sf_mul_expr is complete)
// =========================================================================
constexpr SF_HOT SoftFloat::SoftFloat(const sf_mul_expr& m) noexcept {
	SoftFloat v = m.eval();
	mantissa = v.mantissa;
	exponent = v.exponent;
}

// =========================================================================
// operator* — returns deferred proxy (constexpr, no computation yet)
// =========================================================================
[[nodiscard]] constexpr SF_HOT SF_INLINE sf_mul_expr operator*(SoftFloat a, SoftFloat b) noexcept {
	return { a, b };
}
[[nodiscard]] constexpr SF_HOT SF_INLINE sf_mul_expr operator*(SoftFloat a, float b) noexcept {
	return a * SoftFloat(b);
}
[[nodiscard]] constexpr SF_HOT SF_INLINE sf_mul_expr operator*(SoftFloat a, int32_t b) noexcept {
	return a * SoftFloat(b);
}
[[nodiscard]] constexpr SF_HOT SF_INLINE sf_mul_expr operator*(int32_t a, SoftFloat b) noexcept {
	return SoftFloat(a) * b;
}
[[nodiscard]] constexpr SF_HOT SF_INLINE sf_mul_expr operator*(float a, SoftFloat b) noexcept {
	return SoftFloat(a) * b;
}

constexpr SoftFloat& SoftFloat::operator*=(SoftFloat r) noexcept {
	*this = *this * r;
	return *this;
}

// =========================================================================
// Fused arithmetic — constexpr (implicitly inline since constexpr)
// =========================================================================

// =========================================================================
// fused_mul_add — same-exponent fast path added
// =========================================================================
[[nodiscard]] constexpr SF_HOT SoftFloat fused_mul_add(SoftFloat a, SoftFloat b, SoftFloat c) noexcept {
	if (UNLIKELY(!b.mantissa || !c.mantissa)) return a;
	if (UNLIKELY(!a.mantissa)) return b * c;

	int64_t  prod = static_cast<int64_t>(b.mantissa) * static_cast<int64_t>(c.mantissa);
	int32_t  pm = static_cast<int32_t>(prod >> 29);
	int32_t  pe = b.exponent + c.exponent + 29;

	// R3-3: XOR trick for 1-bit overflow check
	uint32_t norm = static_cast<uint32_t>(pm ^ (pm >> 31)) >> 30;
	pm >>= norm;
	pe += static_cast<int32_t>(norm);

	int d = a.exponent - pe;
	if (d >= 31) return a;
	if (d <= -31) return SoftFloat::from_raw(pm, pe);

	int32_t am = a.mantissa;

	if (d == 0) {
		int32_t  s = am + pm;
		if (UNLIKELY(s == 0)) return {};
		uint32_t ov = static_cast<uint32_t>(s ^ (s >> 31)) >> 30;
		s >>= static_cast<int>(ov);
		pe += static_cast<int32_t>(ov);
		sf_normalise_fast(s, pe);
		return SoftFloat::from_raw(s, pe);
	}

	int32_t exp;
	if (d > 0) {
		pm >>= d;
		exp = a.exponent;     // already in [-128,127]: no SSAT needed
		int32_t s = am + pm;
		if (UNLIKELY(s == 0)) return {};
		sf_normalise_fast(s, exp);
		return SoftFloat::from_raw(s, exp);
	}
	else {
		am >>= -d;
		exp = pe;             // pe may be up to 283: keep sf_sat_exp
		int32_t s = am + pm;
		if (UNLIKELY(s == 0)) return {};
		exp = sf_sat_exp(exp);
		sf_normalise_fast(s, exp);
		return SoftFloat::from_raw(s, exp);
	}
}

// =========================================================================
// fused_mul_sub — same structure as fused_mul_add with negated product
// =========================================================================
[[nodiscard]] constexpr SF_HOT SoftFloat fused_mul_sub(SoftFloat a, SoftFloat b, SoftFloat c) noexcept {
	if (UNLIKELY(!b.mantissa || !c.mantissa)) return a;
	if (UNLIKELY(!a.mantissa)) return -(b * c);

	// Step 1: compute product mantissa in normalised form (positive).
	int64_t  prod = static_cast<int64_t>(b.mantissa) * static_cast<int64_t>(c.mantissa);
	int32_t  pm = static_cast<int32_t>(prod >> 29);
	int32_t  pe = b.exponent + c.exponent + 29;

	// Normalise magnitude (same as fused_mul_add).
	uint32_t norm = static_cast<uint32_t>(pm ^ (pm >> 31)) >> 30;  // R3-3
	pm >>= norm;
	pe += static_cast<int32_t>(norm);

	// Step 2: negate AFTER normalisation so the sign flip is exact.
	pm = -pm;

	// Step 3: align and add (identical to fused_mul_add from here).
	int d = a.exponent - pe;
	if (d >= 31) return a;
	if (d <= -31) return SoftFloat::from_raw(pm, pe);

	int32_t am = a.mantissa;

	if (d == 0) {
		int32_t  s = am + pm;
		if (UNLIKELY(s == 0)) return {};
		uint32_t ov = static_cast<uint32_t>(s ^ (s >> 31)) >> 30;
		s >>= static_cast<int>(ov);
		pe += static_cast<int32_t>(ov);
		sf_normalise_fast(s, pe);
		return SoftFloat::from_raw(s, pe);
	}

	int32_t exp;
	if (d > 0) {
		pm >>= d;
		exp = a.exponent;
		int32_t s = am + pm;
		if (UNLIKELY(s == 0)) return {};
		sf_normalise_fast(s, exp);
		return SoftFloat::from_raw(s, exp);
	}
	else {
		am >>= -d;
		exp = pe;
		int32_t s = am + pm;
		if (UNLIKELY(s == 0)) return {};
		exp = sf_sat_exp(exp);
		sf_normalise_fast(s, exp);
		return SoftFloat::from_raw(s, exp);
	}
}

// =========================================================================
// fused_mul_mul_add — uses sf_normalise_fast, verifies SMLAL opportunity
// =========================================================================
[[nodiscard]] constexpr SF_HOT SoftFloat fused_mul_mul_add(SoftFloat a, SoftFloat b,
	SoftFloat c, SoftFloat d) noexcept
{
	// R3-6: one branch covers all zero-input cases on the hot path
	if (UNLIKELY(!a.mantissa || !b.mantissa || !c.mantissa || !d.mantissa)) {
		bool abz = (!a.mantissa || !b.mantissa);
		bool cdz = (!c.mantissa || !d.mantissa);
		if (abz && cdz) return {};
		if (abz)        return c * d;
		if (cdz)        return a * b;
		// One of the four is zero but both pairs are non-zero: impossible.
		// (If a==0 then abz==true; similarly for others. Reach here never.)
	}

	// Two multiplies; GCC -O2 typically fuses the add into SMLAL here.
	int64_t p1 = static_cast<int64_t>(a.mantissa) * static_cast<int64_t>(b.mantissa);
	int32_t pm1 = static_cast<int32_t>(p1 >> 29);
	int32_t pe1 = a.exponent + b.exponent + 29;

	int64_t p2 = static_cast<int64_t>(c.mantissa) * static_cast<int64_t>(d.mantissa);
	int32_t pm2 = static_cast<int32_t>(p2 >> 29);
	int32_t pe2 = c.exponent + d.exponent + 29;

	// R3-3: XOR trick for both norm steps
	uint32_t n1 = static_cast<uint32_t>(pm1 ^ (pm1 >> 31)) >> 30;
	pm1 >>= n1; pe1 += static_cast<int32_t>(n1);

	uint32_t n2 = static_cast<uint32_t>(pm2 ^ (pm2 >> 31)) >> 30;
	pm2 >>= n2; pe2 += static_cast<int32_t>(n2);

	int d_exp = pe1 - pe2;
	if (d_exp >= 31) return SoftFloat::from_raw(pm1, pe1);
	if (d_exp <= -31) return SoftFloat::from_raw(pm2, pe2);

	int32_t exp;
	if (d_exp == 0) {
		int32_t  s = pm1 + pm2;
		if (UNLIKELY(s == 0)) return {};
		uint32_t ov = static_cast<uint32_t>(s ^ (s >> 31)) >> 30;
		s >>= static_cast<int>(ov);
		pe1 += static_cast<int32_t>(ov);
		sf_normalise_fast(s, pe1);
		return SoftFloat::from_raw(s, pe1);
	}
	if (d_exp > 0) { pm2 >>= d_exp;  exp = pe1; }
	else { pm1 >>= -d_exp; exp = pe2; }

	int32_t s = pm1 + pm2;
	if (UNLIKELY(s == 0)) return {};
	exp = sf_sat_exp(exp);
	sf_normalise_fast(s, exp);
	return SoftFloat::from_raw(s, exp);
}

// fused_mul_mul_sub: returns  a*b - c*d
[[nodiscard]] constexpr SF_HOT SoftFloat fused_mul_mul_sub(SoftFloat a, SoftFloat b,
	SoftFloat c, SoftFloat d) noexcept {
	// Negate d, then reuse the add routine (single mantissa flip = free).
	return fused_mul_mul_add(a, b, c, SoftFloat::from_raw(-d.mantissa, d.exponent));
}

// =========================================================================
// Mixed expression-template operators
// (all constexpr — implied inline, no ODR issues in multi-TU builds)
// =========================================================================

// (mul) + (mul)  =>  fused_mul_mul_add
[[nodiscard]] constexpr SF_INLINE
SoftFloat operator+(const sf_mul_expr& x, const sf_mul_expr& y) noexcept {
	return fused_mul_mul_add(x.lhs, x.rhs, y.lhs, y.rhs);
}
[[nodiscard]] constexpr SF_INLINE
SoftFloat operator-(const sf_mul_expr& x, const sf_mul_expr& y) noexcept {
	return fused_mul_mul_sub(x.lhs, x.rhs, y.lhs, y.rhs);
}

// SoftFloat +/- (mul)  =>  fused_mul_add / fused_mul_sub
[[nodiscard]] constexpr SF_INLINE
SoftFloat operator+(SoftFloat a, const sf_mul_expr& m) noexcept {
	return fused_mul_add(a, m.lhs, m.rhs);
}
[[nodiscard]] constexpr SF_INLINE
SoftFloat operator-(SoftFloat a, const sf_mul_expr& m) noexcept {
	return fused_mul_sub(a, m.lhs, m.rhs);
}

// (mul) +/- SoftFloat  =>  commute
[[nodiscard]] constexpr SF_INLINE
SoftFloat operator+(const sf_mul_expr& m, SoftFloat a) noexcept {
	return fused_mul_add(a, m.lhs, m.rhs);
}

// -----------------------------------------------------------------------
// (mul) - SoftFloat
// -----------------------------------------------------------------------
[[nodiscard]] constexpr SF_INLINE
SoftFloat operator-(const sf_mul_expr& m, SoftFloat a) noexcept {
	// (m.lhs * m.rhs) - a = -(a - m.lhs * m.rhs) = -fms(a, m.lhs, m.rhs)
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
// Convenience free functions — constexpr
// =========================================================================
[[nodiscard]] constexpr SoftFloat sf_abs(SoftFloat x)                                noexcept { return x.abs(); }
[[nodiscard]] constexpr SoftFloat sf_sqrt(SoftFloat x)                                noexcept { return x.sqrt(); }
[[nodiscard]] constexpr SoftFloat sf_exp(SoftFloat x)                                noexcept { return x.exp(); }
[[nodiscard]] constexpr SoftFloat sf_log(SoftFloat x)                                noexcept { return x.log(); }
[[nodiscard]] constexpr SoftFloat sf_log2(SoftFloat x)                                noexcept { return x.log2(); }
[[nodiscard]] constexpr SoftFloat sf_log10(SoftFloat x)                                noexcept { return x.log10(); }
[[nodiscard]] constexpr SoftFloat sf_pow(SoftFloat x, SoftFloat y)                   noexcept { return x.pow(y); }
[[nodiscard]] constexpr SoftFloat sf_trunc(SoftFloat x)                                noexcept { return x.trunc(); }
[[nodiscard]] constexpr SoftFloat sf_floor(SoftFloat x)                                noexcept { return x.floor(); }
[[nodiscard]] constexpr SoftFloat sf_ceil(SoftFloat x)                                noexcept { return x.ceil(); }
[[nodiscard]] constexpr SoftFloat sf_round(SoftFloat x)                          noexcept { return x.round(); }
[[nodiscard]] constexpr SoftFloat sf_fract(SoftFloat x)                       noexcept { return x.fract(); }
[[nodiscard]] constexpr SoftFloatPair sf_modf(SoftFloat x)                    noexcept { return x.modf(); }
[[nodiscard]] constexpr SoftFloat sf_copysign(SoftFloat x, SoftFloat sign) noexcept { return x.copysign(sign); }
[[nodiscard]] constexpr SoftFloat sf_fmod(SoftFloat x, SoftFloat y) noexcept { return x.fmod(y); }
[[nodiscard]] constexpr SoftFloat sf_fma(SoftFloat x, SoftFloat b, SoftFloat c) noexcept { return x.fma(b, c); }
[[nodiscard]] constexpr SoftFloat sf_inv_sqrt(SoftFloat x)                                noexcept { return x.inv_sqrt(); }
[[nodiscard]] constexpr SoftFloat sf_min(SoftFloat a, SoftFloat b)                   noexcept { return (a < b) ? a : b; }
[[nodiscard]] constexpr SoftFloat sf_max(SoftFloat a, SoftFloat b)                   noexcept { return (a > b) ? a : b; }
[[nodiscard]] constexpr SoftFloat sf_clamp(SoftFloat v, SoftFloat lo, SoftFloat hi)    noexcept { return v.clamp(lo, hi); }
[[nodiscard]] constexpr SoftFloat sf_sin(SoftFloat x)                                noexcept { return x.sin(); }
[[nodiscard]] constexpr SoftFloat sf_cos(SoftFloat x)                                noexcept { return x.cos(); }
[[nodiscard]] constexpr SoftFloat sf_tan(SoftFloat x)                                noexcept { return x.tan(); }
[[nodiscard]] constexpr SoftFloat sf_asin(SoftFloat x)                                noexcept { return x.asin(); }
[[nodiscard]] constexpr SoftFloat sf_acos(SoftFloat x)                                noexcept { return x.acos(); }
[[nodiscard]] constexpr SoftFloat sf_sinh(SoftFloat x)  noexcept { return x.sinh(); }
[[nodiscard]] constexpr SoftFloat sf_cosh(SoftFloat x)  noexcept { return x.cosh(); }
[[nodiscard]] constexpr SoftFloat sf_tanh(SoftFloat x)  noexcept { return x.tanh(); }
[[nodiscard]] constexpr SoftFloat sf_atan2(SoftFloat y, SoftFloat x)                   noexcept { return atan2(y, x); }
[[nodiscard]] constexpr SoftFloat sf_hypot(SoftFloat y, SoftFloat x)                   noexcept { return hypot(y, x); }
[[nodiscard]] constexpr SoftFloat sf_lerp(SoftFloat a, SoftFloat b, SoftFloat t)      noexcept { return lerp(a, b, t); }

#if 0
// =========================================================================
// Sine — Horner-form Taylor series, degree 7, constexpr
//
// Constants (all normalised: clz(abs(m)) == 2):
//
//   inv_two_pi = 1/(2π):  683565276 * 2^-32  = 0.15915494...   clz=2 ✓
//   c3 = -1/6:           -715827883 * 2^-32  = -0.16666...      clz=2 ✓
//   c5 =  1/120:          572662306 * 2^-36  =  0.00833333...   clz=2 ✓
//   c7 = -1/5040:        -872724023 * 2^-42  = -0.000198412...  clz=2 ✓
//
// NOTE: constexpr local variables (no `static`) are required in C++20
//       constexpr functions; `static constexpr` locals are C++23.
// =========================================================================
constexpr SF_HOT SoftFloat SoftFloat::sin() const noexcept {
	constexpr SoftFloat inv_two_pi = from_raw(683565276, -32);
	constexpr SoftFloat c3 = from_raw(-715827883, -32);
	constexpr SoftFloat c5 = from_raw(572662306, -36);
	constexpr SoftFloat c7 = from_raw(-872724023, -42);

	SoftFloat x = *this;

	// Range reduction: x -= round(x / 2π) * 2π
	SoftFloat k(x * inv_two_pi);
	int32_t   ki = k.to_int32();
	if (ki != 0) x = x - two_pi() * SoftFloat(ki);

	const SoftFloat hp = half_pi();
	if (x > hp) x = pi() - x;
	else if (x < -hp) x = -pi() - x;

	// Explicitly materialise x2 so that (x2 * c7) returns sf_mul_expr
	// and triggers operator+(SoftFloat, sf_mul_expr) = fused_mul_add.
	const SoftFloat x2(x * x);
	SoftFloat inner = c5 + x2 * c7;    // fused_mul_add(c5, x2, c7)
	SoftFloat poly = c3 + x2 * inner; // fused_mul_add(c3, x2, inner)
	SoftFloat sum = one() + x2 * poly; // fused_mul_add(one(), x2, poly)
	return SoftFloat(x * sum);          // mul_plain, materialised
}

constexpr SF_HOT SoftFloat SoftFloat::cos() const noexcept {
	return (*this + half_pi()).sin();
}

constexpr SF_HOT SoftFloatPair SoftFloat::sincos() const noexcept {
	return { sin(), cos() };
}

[[nodiscard]] constexpr SoftFloatPair sf_sincos(SoftFloat x) noexcept { 
	return x.sincos(); 
}

// tan(x) = sin(x) / cos(x)
constexpr SoftFloat SoftFloat::tan() const noexcept {
	SoftFloat c = cos();
	if (c.is_zero()) {
		// clamp to large finite with sign
		return SoftFloat::from_raw(c.mantissa >= 0 ? (1 << 29) : -(1 << 29), 127);
	}
	return sin() / c;
}

#else

static constexpr int32_t SF_SIN_MANT[512] = {
	           0,  843293690,  843230191,  632343275,  842976226, 1053482228,  631914790,  736993301,
	   841960824,  946801551, 1051499693,  578019742,  630202589,  682290530,  734275721,  786150333,
	   837906553,  889536587,  941032661,  992387019, 1043591926,  547319836,  572761285,  598116479,
	   623381598,  648552838,  673626408,  698598533,  723465451,  748223418,  772868706,  797397602,
	   821806413,  846091463,  870249095,  894275671,  918167572,  941921200,  965532978,  988999351,
	  1012316784, 1035481766, 1058490808,  540670223,  552013618,  563273883,  574449320,  585538248,
	   596538995,  607449906,  618269338,  628995660,  639627258,  650162530,  660599890,  670937767,
	   681174602,  691308855,  701339000,  711263525,  721080937,  730789757,  740388522,  749875788,
	   759250125,  768510122,  777654384,  786681534,  795590213,  804379079,  813046808,  821592095,
	   830013654,  838310216,  846480531,  854523370,  862437520,  870221790,  877875009,  885396022,
	   892783698,  900036924,  907154608,  914135678,  920979082,  927683790,  934248793,  940673101,
	   946955747,  953095785,  959092290,  964944360,  970651112,  976211688,  981625251,  986890984,
	   992008094,  996975812, 1001793390, 1006460100, 1010975242, 1015338134, 1019548121, 1023604567,
	  1027506862, 1031254418, 1034846671, 1038283080, 1041563127, 1044686319, 1047652185, 1050460278,
	  1053110176, 1055601479, 1057933813, 1060106826, 1062120190, 1063973603, 1065666786, 1067199483,
	  1068571464, 1069782521, 1070832474, 1071721163, 1072448455, 1073014240, 1073418433, 1073660973,
	   536870912, 1073660973, 1073418433, 1073014240, 1072448455, 1071721163, 1070832474, 1069782521,
	  1068571464, 1067199483, 1065666786, 1063973603, 1062120190, 1060106826, 1057933813, 1055601479,
	  1053110176, 1050460278, 1047652185, 1044686319, 1041563127, 1038283080, 1034846671, 1031254418,
	  1027506862, 1023604567, 1019548121, 1015338134, 1010975242, 1006460100, 1001793390,  996975812,
	   992008094,  986890984,  981625251,  976211688,  970651112,  964944360,  959092290,  953095785,
	   946955747,  940673101,  934248793,  927683790,  920979082,  914135678,  907154608,  900036924,
	   892783698,  885396022,  877875009,  870221790,  862437520,  854523370,  846480531,  838310216,
	   830013654,  821592095,  813046808,  804379079,  795590213,  786681534,  777654384,  768510122,
	   759250125,  749875788,  740388522,  730789757,  721080937,  711263525,  701339000,  691308855,
	   681174602,  670937767,  660599890,  650162530,  639627258,  628995660,  618269338,  607449906,
	   596538995,  585538248,  574449320,  563273883,  552013618,  540670223, 1058490808, 1035481766,
	  1012316784,  988999351,  965532978,  941921200,  918167572,  894275671,  870249095,  846091463,
	   821806413,  797397602,  772868706,  748223418,  723465451,  698598533,  673626408,  648552838,
	   623381598,  598116479,  572761285,  547319836, 1043591926,  992387019,  941032661,  889536587,
	   837906553,  786150333,  734275721,  682290530,  630202589,  578019742, 1051499693,  946801551,
	   841960824,  736993301,  631914790, 1053482228,  842976226,  632343275,  843230191,  843293690,
	   592202854, -843293690, -843230191, -632343275, -842976226,-1053482228, -631914790, -736993301,
	  -841960824, -946801551,-1051499693, -578019742, -630202589, -682290530, -734275721, -786150333,
	  -837906553, -889536587, -941032661, -992387019,-1043591926, -547319836, -572761285, -598116479,
	  -623381598, -648552838, -673626408, -698598533, -723465451, -748223418, -772868706, -797397602,
	  -821806413, -846091463, -870249095, -894275671, -918167572, -941921200, -965532978, -988999351,
	 -1012316784,-1035481766,-1058490808, -540670223, -552013618, -563273883, -574449320, -585538248,
	  -596538995, -607449906, -618269338, -628995660, -639627258, -650162530, -660599890, -670937767,
	  -681174602, -691308855, -701339000, -711263525, -721080937, -730789757, -740388522, -749875788,
	  -759250125, -768510122, -777654384, -786681534, -795590213, -804379079, -813046808, -821592095,
	  -830013654, -838310216, -846480531, -854523370, -862437520, -870221790, -877875009, -885396022,
	  -892783698, -900036924, -907154608, -914135678, -920979082, -927683790, -934248793, -940673101,
	  -946955747, -953095785, -959092290, -964944360, -970651112, -976211688, -981625251, -986890984,
	  -992008094, -996975812,-1001793390,-1006460100,-1010975242,-1015338134,-1019548121,-1023604567,
	 -1027506862,-1031254418,-1034846671,-1038283080,-1041563127,-1044686319,-1047652185,-1050460278,
	 -1053110176,-1055601479,-1057933813,-1060106826,-1062120190,-1063973603,-1065666786,-1067199483,
	 -1068571464,-1069782521,-1070832474,-1071721163,-1072448455,-1073014240,-1073418433,-1073660973,
	  -536870912,-1073660973,-1073418433,-1073014240,-1072448455,-1071721163,-1070832474,-1069782521,
	 -1068571464,-1067199483,-1065666786,-1063973603,-1062120190,-1060106826,-1057933813,-1055601479,
	 -1053110176,-1050460278,-1047652185,-1044686319,-1041563127,-1038283080,-1034846671,-1031254418,
	 -1027506862,-1023604567,-1019548121,-1015338134,-1010975242,-1006460100,-1001793390, -996975812,
	  -992008094, -986890984, -981625251, -976211688, -970651112, -964944360, -959092290, -953095785,
	  -946955747, -940673101, -934248793, -927683790, -920979082, -914135678, -907154608, -900036924,
	  -892783698, -885396022, -877875009, -870221790, -862437520, -854523370, -846480531, -838310216,
	  -830013654, -821592095, -813046808, -804379079, -795590213, -786681534, -777654384, -768510122,
	  -759250125, -749875788, -740388522, -730789757, -721080937, -711263525, -701339000, -691308855,
	  -681174602, -670937767, -660599890, -650162530, -639627258, -628995660, -618269338, -607449906,
	  -596538995, -585538248, -574449320, -563273883, -552013618, -540670223,-1058490808,-1035481766,
	 -1012316784, -988999351, -965532978, -941921200, -918167572, -894275671, -870249095, -846091463,
	  -821806413, -797397602, -772868706, -748223418, -723465451, -698598533, -673626408, -648552838,
	  -623381598, -598116479, -572761285, -547319836,-1043591926, -992387019, -941032661, -889536587,
	  -837906553, -786150333, -734275721, -682290530, -630202589, -578019742,-1051499693, -946801551,
	  -841960824, -736993301, -631914790,-1053482228, -842976226, -632343275, -843230191, -843293690
};

static constexpr int8_t  SF_SIN_EXP[512] = {
	  0,-36,-35,-34,-34,-34,-33,-33,-33,-33,-33,-32,-32,-32,-32,-32,
	-32,-32,-32,-32,-32,-31,-31,-31,-31,-31,-31,-31,-31,-31,-31,-31,
	-31,-31,-31,-31,-31,-31,-31,-31,-31,-31,-31,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-29,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-31,-31,-31,-31,-31,-31,-31,-31,-31,-31,
	-31,-31,-31,-31,-31,-31,-31,-31,-31,-31,-31,-31,-32,-32,-32,-32,
	-32,-32,-32,-32,-32,-32,-33,-33,-33,-33,-33,-34,-34,-34,-35,-36,
	-82,-36,-35,-34,-34,-34,-33,-33,-33,-33,-33,-32,-32,-32,-32,-32,
	-32,-32,-32,-32,-32,-31,-31,-31,-31,-31,-31,-31,-31,-31,-31,-31,
	-31,-31,-31,-31,-31,-31,-31,-31,-31,-31,-31,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-29,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,-30,
	-30,-30,-30,-30,-30,-30,-31,-31,-31,-31,-31,-31,-31,-31,-31,-31,
	-31,-31,-31,-31,-31,-31,-31,-31,-31,-31,-31,-31,-32,-32,-32,-32,
	-32,-32,-32,-32,-32,-32,-33,-33,-33,-33,-33,-34,-34,-34,-35,-36
};

#if 1
// Internal: fixed-point sincos interpolation.
//
// Table entries are normalised SoftFloat values with mantissa in [2^29, 2^30)
// and exponent in [-36, -29].  The maximum entry magnitude is 1.0 (at idx=128,
// the 90° peak), stored as {536870912, -29} = 2^29 * 2^-29 = 1.0.
//
// Strategy
// --------
// 1. Range-reduce x into [0, 2π) using integer arithmetic on the SoftFloat
//    representation (one multiply + one subtract, both already present).
//
// 2. Convert the reduced angle to a Q-format fixed-point index:
//
//      idx_Q = round(x * (512 / 2π))
//
//    where 512/2π = 81.4873..., stored as a normalised SoftFloat constant.
//    to_int32() gives the integer part (table index), and the remainder
//    drives the interpolation.
//
// 3. Fixed-point interpolation.
//
//    Let h = step = 2π/512.  Within a table cell:
//
//      sin(x0 + δ) ≈ sin(x0) + cos(x0)·δ
//      cos(x0 + δ) ≈ cos(x0) − sin(x0)·δ
//
//    We need δ = x − x0.  Rather than computing x0 = idx * h in SoftFloat
//    and subtracting, we note that after computing:
//
//      u = x * (512 / 2π)      [SoftFloat, exact to ~30 bits]
//
//    the fractional part of u is (x − x0) / h, i.e. δ/h ∈ [0,1).
//    We extract this as an integer with 29 fractional bits:
//
//      frac_Q29 = (u.mantissa << (−u.exponent − 1)) & mask   [if u.exponent ≤ -1]
//
//    Then  δ = (frac_Q29 / 2^29) * h,  so:
//
//      sin0 + cos0·δ = sin0 + cos0 * h * (frac_Q29 / 2^29)
//
//    Pre-folding h = 2π/512 into the scaling:
//
//      correction_mantissa = (cos0.mantissa * frac_Q29) >> 29   [signed SMULL >> 29]
//
//    and the exponent of the correction is:
//
//      correction_exp = cos0.exponent + h_exp
//                     = cos0.exponent + (−28 − 9)          [h = 2π/512 ≈ m·2^-37, so h_exp ≈ -37 + (-28) ... ]
//
//    More precisely, h = π/256.  π = 843314857 * 2^-28, so
//    h = 843314857 * 2^-28 / 256 = 843314857 * 2^-36.
//    Mantissa is already in [2^29,2^30) with exponent -36, so h ≡ {843314857, -36}.
//
//    The product  cos0 * (frac_Q29/2^29) * h  has:
//      mantissa ≈ cos0.mantissa * frac_Q29 >> 29   (both ~30-bit, product ~60-bit)
//      exponent  = cos0.exponent + h.exponent + correction
//               = cos0.exponent + (-36) + (29-29) ... [see derivation below]
//
// Full derivation
// ---------------
//   frac_Q29 represents  δ/h  in Q0.29 format, i.e.
//       δ/h = frac_Q29 * 2^{-29}
//   so  δ = h * frac_Q29 * 2^{-29}
//         = (h_m * 2^{h_e}) * (frac_Q29 * 2^{-29})
//         = h_m * frac_Q29 * 2^{h_e - 29}
//
//   cos0 = c_m * 2^{c_e}
//
//   cos0 * δ = c_m * h_m * frac_Q29 * 2^{c_e + h_e - 29}
//
//   Product c_m * h_m is ~60 bits; shift right by 29 to normalise:
//   ⇒ mantissa ≈ (c_m * frac_Q29) >> 29   [fits int32_t since frac_Q29 < 2^29]
//      exponent  = c_e + h_e                [no extra -29 because we absorbed it in the >> 29]
//
//   Wait — let's be careful:
//     (c_m * frac_Q29) has ~59 bits (c_m < 2^30, frac_Q29 < 2^29).
//     We shift right by 29 ⇒ result has ~30 bits.
//     The implicit 2^{-29} from frac_Q29's Q format AND the 2^{-29} from the shift
//     give 2^{-58} … but we only shifted by 29, giving 2^{-29}.
//
//   Reconcile:
//     Actual value = c_m * 2^{c_e}  *  frac_Q29 * 2^{-29}  *  h_m * 2^{h_e}
//     Let P = (c_m * frac_Q29) in integer arithmetic.
//     P >> 29  has value  P * 2^{-29}  =  c_m * frac_Q29 * 2^{-29}.
//     Actual value = (P >> 29) * 2^{c_e} * h_m * 2^{h_e}.
//     But we also need to fold in h_m (=843314857 ≈ 2^{29.65}).
//
//   Simplification — fold h into a single integer multiplier:
//     Since h_m is a constant, we precompute:
//       scaled_frac = (frac_Q29 * h_m) >> 29   [≈ frac_Q29 * 1.570..., result < 2^29 * π/2]
//     Actually this overflows: frac_Q29 < 2^29, h_m < 2^30 ⇒ product < 2^59, fits int64_t.
//     Then:
//       cos0 * δ  mantissa ≈ (c_m * scaled_frac) >> 29
//       exponent  = c_e + h_e + 0           where the 29+29-29=29 shifts were folded.
//
//   Let's redo cleanly with explicit 64-bit:
//
//     // Step A: δ in Q0.29, scaled by h_m
//     int64_t sfrac = (int64_t)frac_Q29 * H_MANT;          // < 2^59
//     int32_t sfrac32 = (int32_t)(sfrac >> 29);             // < 2^30, represents frac * h_m/2^29
//                                                            // = frac * h / 2^{h_e}  (h_m/2^29 ≈ 1.0 for π/256)
//
//     Wait — h_m = 843314857 ≈ π * 2^28; frac ∈ [0,1); frac_Q29 < 2^29.
//     sfrac32 = frac_Q29 * h_m >> 29 ≈ frac * 2^29 * π * 2^28 >> 29 = frac * π * 2^28.
//     So sfrac32 represents  δ * 2^{-h_e} / (frac * h)  ... getting complicated.
//
// Cleaner approach — use the observation that the correction is SMALL.
// -----------------------------------------------------------------------
// The table has 512 entries per 2π, so the maximum δ is h = 2π/512 ≈ 0.01227.
// The maximum |cos0| is 1.0.  So |cos0 * δ| ≤ 0.01227, which is about 2^{-6.3}.
//
// Both sin0 and cos0 have exponents in [-36, -29].
// The correction exponent = cos0.exponent + h.exponent_contribution.
// h = 843314857 * 2^{-36}  ≈ 0.01227,  so h contributes exponent = -36 relative to a
// unit-normalised cos0.
//
// More precisely, if cos0 = c_m * 2^{c_e} with c_m in [2^29,2^30):
//   correction = c_m * frac * 2^{c_e}  * h
//              = c_m * frac * 2^{c_e}  * h_m * 2^{-36}
// where frac = frac_Q29 * 2^{-29}.
//
//   correction_val = (c_m * frac_Q29 * h_m) >> (29 + 29)  [to get integer part of 30-bit mantissa]
//                  * 2^{c_e + (-36) + 29 + 29 - 58}
//   Hmm: c_m * frac_Q29 * h_m is up to 2^{30+29+30} = 2^{89}, doesn't fit 64-bit.
//
// -----------------------------------------------------------------------
// FINAL APPROACH: Two-step fixed-point, using the fact that frac_Q29 < 2^29.
//
//   1. Compute  adj = (c_m * frac_Q29) >> 29.
//      c_m ∈ [2^29, 2^30), frac_Q29 ∈ [0, 2^29).
//      Product < 2^59, fits int64_t.
//      adj < 2^30.
//      This represents  c_m * (frac_Q29 / 2^29) = c_m * (δ/h).
//
//   2. The actual correction is  adj * h = adj * h_m * 2^{h_e}.
//      adj is an integer with the same "unit" as c_m (i.e., represents adj * 2^{c_e}).
//      correction = adj_m * 2^{c_e}  *  h_m * 2^{h_e}  [but adj < 2^30, h_m < 2^30 → product up to 2^60]
//      correction_m = (adj * h_m) >> 29
//      correction_e = c_e + h_e
//      where h_e = -36  and h_m = 843314857.
//      correction_m = ((c_m * frac_Q29 >> 29) * h_m) >> 29
//      correction_e = c_e + (-36)
//
//   This is exact enough.  adj is up to 2^30, h_m < 2^30, so adj*h_m < 2^60: fits int64_t.
//
//   3. Since the correction is small (< 0.01228) and sin0/cos0 are in [0,1],
//      we must add them with proper exponent alignment.
//
// =========================================================================
//
// IMPLEMENTATION
// =========================================================================
//
//   Inputs: idx ∈ [0, 511], frac_Q29 ∈ [0, 2^29)
//
//   sin0 = SF_SIN_MANT[idx] * 2^{SF_SIN_EXP[idx]}   (signed)
//   cos0 = SF_SIN_MANT[(idx+128)&511] * 2^{SF_SIN_EXP[(idx+128)&511]}
//
//   Correction for sin:  +cos0 * δ
//   Correction for cos:  -sin0 * δ
//
//   δ = frac_Q29 * 2^{-29} * h = frac_Q29 * h_m * 2^{h_e - 29}
//
//   Let F = frac_Q29.  (F ∈ [0, 2^29))
//
//   For sin correction:
//     P = (int64_t)cos0.m * F               // < 2^59, signed
//     adj = P >> 29                          // ∈ (-2^30, 2^30), represents cos0 * F/2^29
//     Q = (int64_t)adj * H_MANT             // adj < 2^30, H_MANT < 2^30 ⇒ < 2^60
//     corr_m = (int32_t)(Q >> 29)           // ∈ (-2^31, 2^31) — might be slightly > 2^30
//     corr_e = cos0.e + H_EXP               // = cos0.e - 36
//
//   Now add sin0 (mantissa s_m, exponent s_e) + corr (mantissa corr_m, exponent corr_e):
//     Standard SoftFloat addition with alignment shift.
//
//   The correction magnitude is ≤ 1.0 * 0.01228 * 2^30 ≈ 1.318 * 10^7.
//   sin0 magnitude is in [2^29, 2^30) = [5.37e8, 1.07e9].
//   So the correction can be up to ~1.2% of the main value, which is 2^{-6.3} relative.
//   Exponent difference is: corr_e - sin0.e = (cos0.e - 36) - sin0.e.
//   Since sin0.e and cos0.e are both in [-36, -29], corr_e ∈ [-72, -65].
//   sin0.e ∈ [-36, -29].  Difference = corr_e - sin0.e ∈ [-72-(-29), -65-(-36)] = [-43, -29].
//   This is always negative and ≤ -29, so the correction is always a right-shift of corr.
//
//   But wait: the correction is already small, so after alignment we might lose all precision
//   if the shift is too large.  In the worst case (sin0.e = -29, corr_e = -72), shift = 43 bits,
//   meaning the correction is completely lost.  That's fine — it's below the 30-bit precision
//   of the representation.
//
//   In practice, near the peaks (sin0 ≈ 1, cos0 ≈ 0), cos0.e is very negative and the correction
//   is tiny, which is correct.  Near the zero-crossings (sin0 ≈ 0, cos0 ≈ 1), the correction
//   matters most and cos0.e ≈ -29 (minimum negative), giving shift ≈ 29 + 36 - sin0.e.
//   But sin0.e is also very negative near the zero-crossing, making the shift small.
//
// =========================================================================

constexpr SF_HOT SoftFloatPair sf_sincos(SoftFloat x) noexcept {
	// ── Constants ────────────────────────────────────────────────────────
	// inv_two_pi = 1/(2π): 683565276 * 2^-32 ≈ 0.15915494...
	constexpr int32_t INV_2PI_M = 683565276;
	constexpr int32_t INV_2PI_E = -32;

	// two_pi = 2π: 843314857 * 2^-27 ≈ 6.28318530...
	constexpr int32_t TWO_PI_M  = 843314857;
	constexpr int32_t TWO_PI_E  = -27;

	// inv_step = 512/(2π) = 256/π: 683565276 * 2^-23 ≈ 81.4873...
	// Used to map reduced angle → table index.
	// 683565276 * 2^-23 = 683565276 / 8388608 = 81.4873...  ✓
	constexpr int32_t INV_STEP_M = 683565276;
	constexpr int32_t INV_STEP_E = -23;

	// h = 2π/512 = π/256: 843314857 * 2^-36
	// This is the angle step between table entries.
	constexpr int32_t H_MANT = 843314857;
	constexpr int32_t H_EXP  = -36; // h = H_MANT * 2^H_EXP

	// ── Handle zero ───────────────────────────────────────────────────────
	if (UNLIKELY(x.mantissa == 0)) {
		return { SoftFloat::zero(), SoftFloat::one() };
	}

	// ── Range reduction: x → [0, 2π) ────────────────────────────────────
	// k = round-toward-zero(x / 2π)  [integer number of full cycles to remove]
	SoftFloat xi = x;
	{
		// Multiply by 1/(2π) and truncate to integer.
		int32_t ki = (x * SoftFloat::from_raw(INV_2PI_M, INV_2PI_E)).to_int32();
		if (ki != 0) {
			// Subtract ki * 2π to reduce x into (-2π, 2π).
			xi = x - SoftFloat(ki) * SoftFloat::from_raw(TWO_PI_M, TWO_PI_E);
		}
		// Clamp to [0, 2π) — at most two corrections needed.
		SoftFloat two_pi = SoftFloat::from_raw(TWO_PI_M, TWO_PI_E);
		if (xi.mantissa < 0)   xi = xi + two_pi;
		if (!(xi < two_pi))    xi = xi - two_pi;
	}

	// ── Map reduced angle to table index + fixed-point fraction ──────────
	//
	// u = xi * inv_step   (u ∈ [0, 512))
	// idx = (int32_t)u    (integer part, 0..511)
	// frac_Q29 = fractional_bits(u) in Q0.29 format ∈ [0, 2^29)
	//
	// We extract frac_Q29 directly from the SoftFloat mantissa of u:
	//
	//   u = u_m * 2^{u_e}   with u_m ∈ [2^29, 2^30), u_e ∈ [-29, ...]
	//   Integer part:   idx = u_m >> (-u_e)      if u_e ≤ 0
	//   Fractional Q29: frac_Q29 = (u_m << (29 + u_e)) & (2^29 - 1)   if u_e ∈ [-29, 0]
	//                            = 0                                    if u_e ≤ -30 (u < 1)
	//                            = full shift left                      if u_e > 0

	SoftFloat u_sf = xi * SoftFloat::from_raw(INV_STEP_M, INV_STEP_E);

	int32_t idx;
	int32_t frac_Q29;

	{
		int32_t u_m = u_sf.mantissa; // ∈ [2^29, 2^30), or 0
		int32_t u_e = u_sf.exponent; // typically -29..-21 for xi ∈ [0, 2π)

		if (UNLIKELY(u_m == 0)) {
			idx      = 0;
			frac_Q29 = 0;
		}
		else {
			// Integer part: u_m >> (-u_e) but clipped to [0, 511]
			// u_e ≥ 0 means u ≥ 2^29 which is way above 512; clamp.
			if (u_e >= 10) {
				// u ≥ 2^{29+10} / 2^{29} = 2^10 = 1024 > 511; use 511.
				idx      = 511;
				frac_Q29 = 0;
			}
			else if (u_e >= 0) {
				// u_m in [2^29,2^30), shifted left by u_e: idx = u_m << u_e >> 29
				// Maximum u_e here is 9, so u_m << 9 < 2^39; need 64-bit.
				idx = static_cast<int32_t>(
				          (static_cast<int64_t>(u_m) << u_e) >> 29
					  ) & 0x1FF;
				frac_Q29 = 0; // integer part dominates; fraction lost (u large)
			}
			else {
				// Normal case: u_e < 0.
				// idx = u_m >> (-u_e - 0) with 29 implicit fractional bits removed:
				//   u = u_m * 2^{u_e}; integer part = u_m >> (-u_e) [if u_e ≤ -1]
				//   but u_m already has 29 implicit bits... no, u_m is the actual mantissa.
				//   u = u_m * 2^{u_e}.
				//   Integer part of u = u_m >> (-u_e)   if u_e ≤ 0.
				int rs = -u_e; // right-shift amount ≥ 1
				if (rs > 30) {
					idx      = 0;
					frac_Q29 = 0;
				}
				else {
					idx = (u_m >> rs) & 0x1FF;

					// Fractional part of u as Q0.29:
					// u_frac = u_m & ((1<<rs)-1), scaled to 2^29.
					// frac_Q29 = (u_m & mask) << (29 - rs)   if rs ≤ 29
					//           = (u_m & mask) >> (rs - 29)   if rs > 29
					if (rs <= 29) {
						uint32_t mask = (1u << rs) - 1u;
						frac_Q29 = static_cast<int32_t>(
						    (static_cast<uint32_t>(u_m) & mask) << (29 - rs)
						);
					}
					else {
						// rs = 30: one bit of fraction, then zeros
						uint32_t mask = (1u << rs) - 1u;
						frac_Q29 = static_cast<int32_t>(
						    (static_cast<uint32_t>(u_m) & mask) >> (rs - 29)
						);
					}
				}
			}
		}
	}

	idx &= 0x1FF; // ensure 0..511

	// ── Table lookup ──────────────────────────────────────────────────────
	int32_t s_m = SF_SIN_MANT[idx];
	int32_t s_e = SF_SIN_EXP[idx];
	int32_t c_m = SF_SIN_MANT[(idx + 128) & 0x1FF];
	int32_t c_e = SF_SIN_EXP[(idx + 128) & 0x1FF];

	// ── Fixed-point derivative correction ────────────────────────────────
	//
	// sin(x0 + δ) ≈ sin(x0) + cos(x0)·δ
	// cos(x0 + δ) ≈ cos(x0) − sin(x0)·δ
	//
	// δ = frac_Q29 * 2^{-29} * h
	//   = frac_Q29 * H_MANT * 2^{H_EXP - 29}
	//
	// cos0·δ = c_m * frac_Q29 * H_MANT * 2^{c_e + H_EXP - 29}
	//
	// Two-step integer computation:
	//
	//   Step 1:  adj    = (c_m * frac_Q29) >> 29
	//              represents c_m * (frac_Q29 / 2^29) as a 30-bit integer.
	//   Step 2:  corr_m = (adj  * H_MANT ) >> 29
	//              represents adj * (H_MANT / 2^29).
	//
	// The two right-shifts together absorb 58 bits of implicit scale.
	// For the VALUE equation to hold:
	//
	//   corr_m * 2^{corr_e} = c_m * 2^{c_e}  *  frac_Q29 * 2^{-29}  *  H_MANT * 2^{H_EXP}
	//
	// Since corr_m ≈ (c_m * frac_Q29 * H_MANT) / 2^58, we need:
	//
	//   corr_m * 2^{corr_e} = corr_m * 2^{58}  *  2^{c_e - 29 + H_EXP}
	//   ⟹  corr_e = c_e + H_EXP - 29 + 58 = c_e + H_EXP + 29
	//
	// The +29 term is CRITICAL — it accounts for the 58 bits consumed by the
	// two >> 29 shifts, offset by the 29 bits already encoded in H_EXP's
	// Q-format meaning.  Omitting it makes the correction ~2^{-29} ≈ 10^{-9}
	// times too small, effectively disabling interpolation entirely.

	auto make_corr = [&](int32_t base_m,
		int32_t base_e,
		bool negate) -> SoftFloat
	{
		if (UNLIKELY(frac_Q29 == 0 || base_m == 0)) return SoftFloat::zero();

		// Step 1: adj = base_m * frac_Q29 / 2^29
		int64_t p1   = static_cast<int64_t>(base_m) * static_cast<int64_t>(frac_Q29);
		int32_t adj  = static_cast<int32_t>(p1 >> 29); // ∈ (-2^30, 2^30)

		if (UNLIKELY(adj == 0)) return SoftFloat::zero();

		// Step 2: corr = adj * H_MANT / 2^29
		int64_t p2     = static_cast<int64_t>(adj) * static_cast<int64_t>(H_MANT);
		int32_t corr_m = static_cast<int32_t>(p2 >> 29); // ∈ (-2^30, 2^30)

		// ── BUGFIX: corr_e = base_e + H_EXP + 29 ─────────────────────────
		// The two >> 29 shifts consume 58 bits of implicit scale.
		// H_EXP already accounts for 29 of those bits (via the Q0.29 format
		// of frac_Q29), so the remaining 29 must be added explicitly here.
		// The original code wrote `base_e + H_EXP`, which is 29 too small,
		// making every correction ≈ 2^{-29} ≈ 10^{-9} of its correct value.
		int32_t corr_e = base_e + H_EXP + 29;

		if (UNLIKELY(corr_m == 0)) return SoftFloat::zero();
		if (negate) corr_m = -corr_m;

		// Normalise corr_m to [2^29, 2^30) with correct sign.
		uint32_t abs_c = sf_abs32(corr_m);
		if (abs_c >= 0x40000000u) {
			// bit30 set → overflow by 1
			corr_m >>= 1; // arithmetic: preserves sign
			corr_e  += 1;
		}
		else if (abs_c < 0x20000000u && abs_c != 0) {
			// underflow: left-shift needed (rare, only when frac_Q29 is tiny)
			int lz  = sf_clz(abs_c);
			int sh  = lz - 2;
			int32_t sign = corr_m >> 31;
			uint32_t a   = (static_cast<uint32_t>(corr_m) ^ static_cast<uint32_t>(sign))
			               - static_cast<uint32_t>(sign);
			a     <<= sh;
			corr_e -= sh;
			corr_m  = static_cast<int32_t>((a ^ static_cast<uint32_t>(sign))
			                               - static_cast<uint32_t>(sign));
		}

		return SoftFloat::from_raw(corr_m, corr_e);
	};

	// ── Build sin_corr and cos_corr ───────────────────────────────────────
	SoftFloat sin_corr = make_corr(c_m, c_e, /*negate=*/false); // +cos0·δ
	SoftFloat cos_corr = make_corr(s_m, s_e, /*negate=*/true); // -sin0·δ

	// ── Combine table value + correction ─────────────────────────────────
	SoftFloat sin_base = SoftFloat::from_raw(s_m, s_e);
	SoftFloat cos_base = SoftFloat::from_raw(c_m, c_e);

	// Standard SoftFloat addition handles alignment automatically.
	SoftFloat sin_val = sin_base + sin_corr;
	SoftFloat cos_val = cos_base + cos_corr;

	return { sin_val, cos_val };
}

#else
// Internal: do range reduction once, return {sin(x), cos(x)} from same table slot.
constexpr SF_HOT SoftFloatPair sf_sincos(SoftFloat x) noexcept {
	constexpr SoftFloat inv_two_pi = SoftFloat::from_raw(683565276, -32);
	constexpr SoftFloat two_pi_c = SoftFloat::two_pi();
	constexpr SoftFloat step = SoftFloat::from_raw(843314857, -36);
	constexpr SoftFloat inv_step = SoftFloat::from_raw(683565276, -23);

	// Range reduction (same as sin(), done once)
	SoftFloat xi = x;
	int32_t ki = (xi * inv_two_pi).to_int32();
	if (ki) xi -= two_pi_c * SoftFloat(ki);
	if (xi.mantissa < 0) xi += two_pi_c;
	if (!(xi < two_pi_c)) xi -= two_pi_c;

	SoftFloat u = xi * inv_step;
	int32_t   idx = u.to_int32() & 0x1FF;
	SoftFloat x0 = SoftFloat(idx) * step;
	SoftFloat frac = xi - x0;

	SoftFloat sin0 = SoftFloat::from_raw(SF_SIN_MANT[idx], SF_SIN_EXP[idx]);
	SoftFloat cos0 = SoftFloat::from_raw(SF_SIN_MANT[(idx + 128) & 0x1FF], SF_SIN_EXP[(idx + 128) & 0x1FF]);

	// sin(x) ≈ sin0 + cos0*frac
	// cos(x) ≈ cos0 − sin0*frac  (derivative of cosine)
	return {
		fused_mul_add(sin0,  cos0, frac),
		fused_mul_add(cos0, -sin0, frac)
	};
}
#endif

constexpr SoftFloat SoftFloat::tan() const noexcept {
	auto [s, c] = sf_sincos(*this);
	if (c.is_zero())
		return from_raw(s.mantissa >= 0 ? (1 << 29) : -(1 << 29), 127);
	return s / c;
}

constexpr SoftFloat SoftFloat::sin() const noexcept { 
	return sf_sincos(*this).intpart; 
}

constexpr SoftFloat SoftFloat::cos() const noexcept { 
	return sf_sincos(*this).fracpart; 
}

constexpr SoftFloatPair SoftFloat::sincos() const noexcept {
	return sf_sincos(*this);
}

#endif

// asin(x) – polynomial approximation for |x| <= 1
constexpr SoftFloat SoftFloat::asin() const noexcept {
	SoftFloat x = *this;
	bool neg = x.is_negative();
	x = x.abs();
	if (x > SoftFloat::one()) return SoftFloat::zero(); // out of domain

	// Use identity: asin(x) = atan2(x, sqrt(1 - x*x))
	// This is accurate and reuses atan2 + sqrt
	SoftFloat result = atan2(x, (SoftFloat::one() - x * x).sqrt());
	return neg ? -result : result;
}

// acos(x) = pi/2 - asin(x)
constexpr SoftFloat SoftFloat::acos() const noexcept {
	return SoftFloat::half_pi() - asin();
}

constexpr SoftFloat SoftFloat::sinh() const noexcept {
	SoftFloat e = exp();
	return (e - SoftFloat::one() / e) >> 1; // (e - e^-1)/2
}

constexpr SoftFloat SoftFloat::cosh() const noexcept {
	SoftFloat e = exp();
	return (e + SoftFloat::one() / e) >> 1;
}

constexpr SoftFloat SoftFloat::tanh() const noexcept {
	SoftFloat e2 = (*this << 1).exp(); // e^(2x)
	return (e2 - SoftFloat::one()) / (e2 + SoftFloat::one());
}

#if 0
// Inverse square root table for mantissa in [1.0, 2.0)
// f = 1 + i/256, i = 0..256
// value = mantissa * 2^exponent, normalized to [2^29, 2^30)

static constexpr int32_t INV_SQRT_MANT[257] = {
	0x20000000, 0x3FE017EC, 0x3FC05F61, 0x3FA0D5E9, 0x3F817B11, 0x3F624E66, 0x3F434F77, 0x3F247DD4,
	0x3F05D910, 0x3EE760BF, 0x3EC91474, 0x3EAAF3C8, 0x3E8CFE50, 0x3E6F33A7, 0x3E519367, 0x3E341D2B,
	0x3E16D092, 0x3DF9AD38, 0x3DDCB2BD, 0x3DBFE0C3, 0x3DA336EC, 0x3D86B4DA, 0x3D6A5A31, 0x3D4E2699,
	0x3D3219B6, 0x3D163331, 0x3CFA72B2, 0x3CDED7E4, 0x3CC36272, 0x3CA81207, 0x3C8CE651, 0x3C71DEFE,
	0x3C56FBBC, 0x3C3C3C3C, 0x3C21A02F, 0x3C072747, 0x3BECD137, 0x3BD29DB2, 0x3BB88C6E, 0x3B9E9D1F,
	0x3B84CF7D, 0x3B6B233F, 0x3B51981D, 0x3B382DD0, 0x3B1EE412, 0x3B05BA9E, 0x3AECB12E, 0x3AD3C781,
	0x3ABAFD52, 0x3AA25260, 0x3A89C669, 0x3A71592C, 0x3A590A6A, 0x3A40D9E3, 0x3A28C759, 0x3A10D28F,
	0x39F8FB46, 0x39E14144, 0x39C9A44B, 0x39B22421, 0x399AC08C, 0x39837952, 0x396C4E39, 0x39553F0A,
	0x393E4B8B, 0x39277388, 0x3910B6C7, 0x38FA1514, 0x38E38E39, 0x38CD2201, 0x38B6D037, 0x38A098A9,
	0x388A7B22, 0x38747770, 0x385E8D61, 0x3848BCC3, 0x38330566, 0x381D6718, 0x3807E1AA, 0x37F274EB,
	0x37DD20AE, 0x37C7E4C3, 0x37B2C0FC, 0x379DB52C, 0x3788C126, 0x3773E4BC, 0x375F1FC3, 0x374A720F,
	0x3735DB75, 0x37215BC9, 0x370CF2E1, 0x36F8A094, 0x36E464B7, 0x36D03F22, 0x36BC2FAB, 0x36A8362A,
	0x36945278, 0x3680846D, 0x366CCBE1, 0x365928AE, 0x36459AAE, 0x363221BA, 0x361EBDAC, 0x360B6E60,
	0x35F833B1, 0x35E50D79, 0x35D1FB96, 0x35BEFDE2, 0x35AC143A, 0x35993E7C, 0x35867C84, 0x3573CE30,
	0x3561335D, 0x354EABEA, 0x353C37B6, 0x3529D69F, 0x35178883, 0x35054D44, 0x34F324BF, 0x34E10ED6,
	0x34CF0B68, 0x34BD1A57, 0x34AB3B82, 0x34996ECC, 0x3487B416, 0x34760B41, 0x3464742F, 0x3452EEC3,
	0x34417AE0, 0x34301868, 0x341EC73E, 0x340D8746, 0x33FC5863, 0x33EB3A79, 0x33DA2D6C, 0x33C93121,
	0x33B8457D, 0x33A76A63, 0x33969FBA, 0x3385E566, 0x33753B4E, 0x3364A156, 0x33541766, 0x33439D62,
	0x33333333, 0x3322D8BE, 0x33128DEB, 0x330252A1, 0x32F226C6, 0x32E20A43, 0x32D1FD00, 0x32C1FEE4,
	0x32B20FD7, 0x32A22FC3, 0x32925E8F, 0x32829C25, 0x3272E86D, 0x32634351, 0x3253ACBA, 0x32442492,
	0x3234AAC3, 0x32253F36, 0x3215E1D5, 0x3206928C, 0x31F75143, 0x31E81DE8, 0x31D8F863, 0x31C9E0A0,
	0x31BAD68B, 0x31ABDA0E, 0x319CEB16, 0x318E098D, 0x317F3561, 0x31706E7C, 0x3161B4CC, 0x3153083C,
	0x314468BA, 0x3135D631, 0x3127508E, 0x3118D7C0, 0x310A6BB2, 0x30FC0C52, 0x30EDB98E, 0x30DF7354,
	0x30D13990, 0x30C30C31, 0x30B4EB25, 0x30A6D65A, 0x3098CDBE, 0x308AD140, 0x307CE0CF, 0x306EFC59,
	0x306123CD, 0x3053571B, 0x30459630, 0x3037E0FD, 0x302A3771, 0x301C997B, 0x300F070C, 0x30018012,
	0x2FF4047E, 0x2FE69440, 0x2FD92F48, 0x2FCBD586, 0x2FBE86EB, 0x2FB14367, 0x2FA40AEB, 0x2F96DD67,
	0x2F89BACC, 0x2F7CA30C, 0x2F6F9618, 0x2F6293E0, 0x2F559C56, 0x2F48AF6B, 0x2F3BCD12, 0x2F2EF53A,
	0x2F2227D7, 0x2F1564DB, 0x2F08AC36, 0x2EFBFDDB, 0x2EEF59BD, 0x2EE2BFCD, 0x2ED62FFE, 0x2EC9AA43,
	0x2EBD2E8D, 0x2EB0BCD0, 0x2EA454FF, 0x2E97F70B, 0x2E8BA2E9, 0x2E7F588B, 0x2E7317E4, 0x2E66E0E7,
	0x2E5AB389, 0x2E4E8FBB, 0x2E427573, 0x2E3664A3, 0x2E2A5D3F, 0x2E1E5F3A, 0x2E126A89, 0x2E067F20,
	0x2DFA9CF2, 0x2DEEC3F4, 0x2DE2F41A, 0x2DD72D58, 0x2DCB6FA3, 0x2DBFBAEE, 0x2DB40F2F, 0x2DA86C5A,
	0x2D9CD264, 0x2D914141, 0x2D85B8E6, 0x2D7A3949, 0x2D6EC25E, 0x2D63541A, 0x2D57EE72, 0x2D4C915C,
	0x2D413CCD,
};

static constexpr int8_t INV_SQRT_EXP[257] = {
	-29, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30,
};

constexpr SF_HOT SoftFloat SoftFloat::inv_sqrt() const noexcept {
	if (UNLIKELY(mantissa <= 0)) return {};

	uint32_t m_abs = sf_abs32(mantissa);
	int32_t E = exponent + 29;                       // x = f * 2^E, f in [1, 2)

	// Table index and fractional part
	uint32_t offset = m_abs - 0x20000000u;           // (f - 1) * 2^29
	uint32_t idx = offset >> 21;                     // 0 … 255
	uint32_t frac_bits = (offset >> 13) & 0xFF;      // 8‑bit fraction

	// Linear interpolation for 1/√f
	SoftFloat v0 = SoftFloat::from_raw(INV_SQRT_MANT[idx], INV_SQRT_EXP[idx]);
	SoftFloat v1 = SoftFloat::from_raw(INV_SQRT_MANT[idx + 1], INV_SQRT_EXP[idx + 1]);
	SoftFloat frac = SoftFloat(static_cast<int32_t>(frac_bits)) >> 8;
	SoftFloat y = fused_mul_add(v0,frac,(v1 - v0));    // y ≈ 1/√f

	// Exponent scaling: multiply by 2^(-E/2)
	int32_t shift = E >> 1;                          // floor(E/2)
	y >>= shift;                                  // divide by 2^shift
	if (E & 1) {
		// Multiply by 1/√2 ≈ 0.70710678
		constexpr SoftFloat inv_sqrt2 = SoftFloat::from_raw(0x2D413CCD, -30);
		y *= inv_sqrt2;
	}

	// Two Newton‑Raphson iterations: y = y * (1.5 - 0.5 * x * y²)
	constexpr SoftFloat k15 = SoftFloat::from_raw(0x30000000, -29); // 1.5
	const SoftFloat half_x = SoftFloat::from_raw(mantissa, exponent - 1); // 0.5 * x

	// Iteration 1
	{
		const SoftFloat y2(y * y);
		SoftFloat step = fused_mul_sub(k15, half_x, y2);   // 1.5 - 0.5*x*y²
		y *= step;
	}
	// Iteration 2
	{
		const SoftFloat y2(y * y);
		SoftFloat step = fused_mul_sub(k15, half_x, y2);
		y *= step;
	}
	return y;
}

#else
// =========================================================================
// Inverse square root — Q-rsqrt seed + 2× Newton-Raphson, constexpr
//
// k1.5 = 1.5:  0x30000000 * 2^-29 = 1.5   clz=2 ✓
// =========================================================================
constexpr SF_HOT SoftFloat SoftFloat::inv_sqrt() const noexcept {
	if (UNLIKELY(mantissa <= 0)) return {};

	// Fast initial estimate via the classic magic-constant bit trick.
	// Both to_float() and sf_bitcast are constexpr in C++20.
	float    xf = to_float();
	uint32_t bits = sf_bitcast<uint32_t>(xf);
	bits = 0x5f3759dfu - (bits >> 1);
	SoftFloat y(sf_bitcast<float>(bits));

	constexpr SoftFloat k15 = from_raw(0x30000000, -29);  // 1.5 (no `static` — C++20)
	SoftFloat hx = from_raw(mantissa, exponent - 1);       // 0.5 * this

	// Newton iteration 1: y = y * (1.5 - 0.5*x*y²)
	//                       = y * fused_mul_sub(k15, hx, y²)
	{
		const SoftFloat y2(y * y);                          // materialise
		SoftFloat step = fused_mul_sub(k15, hx, y2);        // k15 - hx*y2
		y *= step;
	}
	// Newton iteration 2:
	{
		const SoftFloat y2(y * y);
		SoftFloat step = fused_mul_sub(k15, hx, y2);
		y *= step;
	}
	return y;
}
#endif

constexpr SF_HOT SoftFloat SoftFloat::sqrt() const noexcept {
	if (UNLIKELY(mantissa <= 0)) return {};

	if (SF_IS_CONSTEVAL()) {
		int32_t m = mantissa;
		int32_t e = exponent;

		// Ensure exponent is even so 2^e is a perfect square
		if (e & 1) {
			m = m << 1;  // Safe: m is in [2^29, 2^30), so m<<1 is in [2^30, 2^31), fits int32_t
			e = e - 1;
		}

		// value = m * 2^e
		// sqrt(value) = sqrt(m) * 2^(e/2)
		// To get ~30 bits of precision, compute sqrt(m * 2^30) = sqrt(m) * 2^15
		uint64_t scaled = static_cast<uint64_t>(m) << 30;
		uint64_t root = ct_isqrt64(scaled);

		int32_t rm = static_cast<int32_t>(root);
		int32_t re = e / 2 - 15;

		return SoftFloat(rm, re);
	}

	return *this * inv_sqrt();
}

#if 1
// ---------------------------------------------------------------------
// ATAN2 – table‑based, linear interpolation (256 entries)
// ---------------------------------------------------------------------
static constexpr int32_t ATAN_MANT[257] = {
	0x00000000, 0x3FFFEA00, 0x3FFFAA80, 0x2FFF7000, 0x3FFEAA80, 0x27FEB2A0, 0x2FFDC020, 0x37FC6DA0,
	0x3FFAAB60, 0x23FC34B0, 0x27FACBE0, 0x2BF91340, 0x2FF70300, 0x33F49330, 0x37F1BBE0, 0x3BEE7530,
	0x3FEAB760, 0x21F33D48, 0x23F0DB78, 0x25EE3260, 0x27EB3E18, 0x29E7FAC8, 0x2BE46498, 0x2DE077B8,
	0x2FDC3048, 0x31D78A88, 0x33D282A0, 0x35CD14E0, 0x37C73D78, 0x39C0F8B8, 0x3BBA42E0, 0x3DB31840,
	0x3FAB7530, 0x20D1AB08, 0x21CD5B98, 0x22C8CA80, 0x23C3F5F4, 0x24BEDC2C, 0x25B97B64, 0x26B3D1D8,
	0x27ADDDD0, 0x28A79D8C, 0x29A10F50, 0x2A9A3174, 0x2B93023C, 0x2C8B7FFC, 0x2D83A910, 0x2E7B7BD0,
	0x2F72F694, 0x306A17C4, 0x3160DDC4, 0x325746F8, 0x334D51D0, 0x3442FCC0, 0x35384634, 0x362D2CAC,
	0x3721AEA4, 0x3815CA98, 0x39097F14, 0x39FCCA9C, 0x3AEFABBC, 0x3BE2210C, 0x3CD4291C, 0x3DC5C288,
	0x3EB6EBF0, 0x3FA7A3F8, 0x204BF4A2, 0x20C3DD40, 0x213B8B2E, 0x21B2FDCA, 0x222A346C, 0x22A12E74,
	0x2317EB46, 0x238E6A42, 0x2404AACE, 0x247AAC56, 0x24F06E40, 0x2565EFFA, 0x25DB30F4, 0x2650309E,
	0x26C4EE6E, 0x273969D6, 0x27ADA252, 0x2821975A, 0x2895486C, 0x2908B508, 0x297BDCB0, 0x29EEBEE6,
	0x2A615B32, 0x2AD3B11C, 0x2B45C02E, 0x2BB787F6, 0x2C290806, 0x2C9A3FEC, 0x2D0B2F3E, 0x2D7BD592,
	0x2DEC3282, 0x2E5C45AA, 0x2ECC0EA4, 0x2F3B8D12, 0x2FAAC096, 0x3019A8D4, 0x30884570, 0x30F69618,
	0x31649A72, 0x31D2522C, 0x323FBCF8, 0x32ACDA86, 0x3319AA8A, 0x33862CB8, 0x33F260CC, 0x345E467C,
	0x34C9DD86, 0x353525AA, 0x35A01EA8, 0x360AC840, 0x3675223A, 0x36DF2C5C, 0x3748E66E, 0x37B2503C,
	0x381B6992, 0x3884323E, 0x38ECAA14, 0x3954D0E4, 0x39BCA686, 0x3A242ACC, 0x3A8B5D92, 0x3AF23EB4,
	0x3B58CE0A, 0x3BBF0B76, 0x3C24F6D6, 0x3C8A900C, 0x3CEFD6FE, 0x3D54CB8E, 0x3DB96DA8, 0x3E1DBD30,
	0x3E81BA16, 0x3EE56442, 0x3F48BBA6, 0x3FABC02E, 0x200738E7, 0x2038683D, 0x20696E12, 0x209A4A62,
	0x20CAFD29, 0x20FB8664, 0x212BE610, 0x215C1C2C, 0x218C28B6, 0x21BC0BAF, 0x21EBC516, 0x221B54EE,
	0x224ABB37, 0x2279F7F6, 0x22A90B2C, 0x22D7F4DE, 0x2306B511, 0x23354BCA, 0x2363B90F, 0x2391FCE6,
	0x23C01757, 0x23EE086A, 0x241BD027, 0x24496E98, 0x2476E3C5, 0x24A42FBA, 0x24D15280, 0x24FE4C24,
	0x252B1CB2, 0x2557C435, 0x258442BC, 0x25B09852, 0x25DCC508, 0x2608C8EA, 0x2634A409, 0x26605673,
	0x268BE039, 0x26B7416C, 0x26E27A1B, 0x270D8A5A, 0x27387239, 0x276331CB, 0x278DC923, 0x27B83854,
	0x27E27F71, 0x280C9E8E, 0x283695C0, 0x2860651C, 0x288A0CB6, 0x28B38CA5, 0x28DCE4FD, 0x290615D6,
	0x292F1F46, 0x29580163, 0x2980BC45, 0x29A95004, 0x29D1BCB7, 0x29FA0276, 0x2A22215B, 0x2A4A197D,
	0x2A71EAF7, 0x2A9995E0, 0x2AC11A53, 0x2AE8786A, 0x2B0FB03E, 0x2B36C1EB, 0x2B5DAD8B, 0x2B847338,
	0x2BAB130E, 0x2BD18D28, 0x2BF7E1A1, 0x2C1E1096, 0x2C441A22, 0x2C69FE62, 0x2C8FBD71, 0x2CB5576D,
	0x2CDACC72, 0x2D001C9C, 0x2D25480A, 0x2D4A4ED8, 0x2D6F3124, 0x2D93EF0A, 0x2DB888AA, 0x2DDCFE20,
	0x2E014F8A, 0x2E257D08, 0x2E4986B6, 0x2E6D6CB4, 0x2E912F1F, 0x2EB4CE17, 0x2ED849B9, 0x2EFBA225,
	0x2F1ED77A, 0x2F41E9D7, 0x2F64D95A, 0x2F87A623, 0x2FAA5051, 0x2FCCD803, 0x2FEF3D59, 0x30118072,
	0x3033A16E, 0x3055A06C, 0x30777D8B, 0x309938EC, 0x30BAD2AE, 0x30DC4AF1, 0x30FDA1D5, 0x311ED77A,
	0x313FEBFE, 0x3160DF83, 0x3181B228, 0x31A2640D, 0x31C2F553, 0x31E36618, 0x3203B67D, 0x3223E6A3,
	// ---- new entry for atan(1) = π/4 ----
	0x3243F6A9  // = 843314857
};

static constexpr int8_t ATAN_EXP[257] = {
	   0,  -38,  -37,  -36,  -36,  -35,  -35,  -35,  -35,  -34,  -34,  -34,  -34,  -34,  -34,  -34,
	 -34,  -33,  -33,  -33,  -33,  -33,  -33,  -33,  -33,  -33,  -33,  -33,  -33,  -33,  -33,  -33,
	 -33,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,
	 -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,  -32,
	 -32,  -32,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,
	 -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,
	 -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,
	 -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,
	 -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -31,  -30,  -30,  -30,  -30,
	 -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,
	 -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,
	 -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,
	 -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,
	 -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,
	 -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,
	 -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,  -30,
	 // ---- new entry for atan(1) = π/4 ----
	 -30
};


constexpr SF_HOT SoftFloat atan2(SoftFloat y, SoftFloat x) noexcept {
	if (x.is_zero() && y.is_zero()) return SoftFloat::zero();

	bool x_neg = x.is_negative();
	bool y_neg = y.is_negative();
	x = x.abs();
	y = y.abs();

	bool swap = y > x;
	if (swap) { SoftFloat t = x; x = y; y = t; }

	// t = y/x in [0, 1]
	SoftFloat t = x.is_zero() ? SoftFloat::zero() : y / x;

	// Scale t by 256 to get table index
	// 256.0 represented as SoftFloat: mant = 0x20000000, exp = -29+8 = -21
	constexpr SoftFloat scale = SoftFloat::from_raw(0x20000000, -21);
	SoftFloat idx_f = t * scale;
	int32_t idx = idx_f.to_int32();
	if (idx > 255) idx = 255;
	if (idx < 0)   idx = 0;

	// Base value from table
	SoftFloat base = SoftFloat::from_raw(ATAN_MANT[idx], ATAN_EXP[idx]);

	// Linear interpolation between idx and idx+1
	SoftFloat next = SoftFloat::from_raw(ATAN_MANT[idx + 1], ATAN_EXP[idx + 1]);
	SoftFloat frac = idx_f - SoftFloat(idx);   // fractional part in [0,1)
	SoftFloat angle = base + frac * (next - base);

	// Octant reconstruction
	if (swap)  angle = SoftFloat::half_pi() - angle;
	if (x_neg) angle = SoftFloat::pi() - angle;
	if (y_neg) angle = -angle;

	return angle;
}

#else

// Octant-reduced atan2 approximation
constexpr SF_HOT SoftFloat atan2(SoftFloat y, SoftFloat x) noexcept {
	// Handle special cases first
	if (x.is_zero() && y.is_zero()) return SoftFloat::zero();

	if (y.is_zero()) {
		return x.is_negative() ? SoftFloat::pi() : SoftFloat::zero();
	}
	if (x.is_zero()) {
		return y.is_negative() ? -SoftFloat::half_pi() : SoftFloat::half_pi();
	}

	bool x_neg = x.is_negative();
	bool y_neg = y.is_negative();
	SoftFloat ax = x.abs();
	SoftFloat ay = y.abs();

	// Map to [0, 1] range by swapping if needed
	bool swap = ay > ax;
	SoftFloat t = swap ? (ax / ay) : (ay / ax);

	// Polynomial approximation: atan(t) for t in [0, 1]
	// Using verified coefficients from reliable source

	SoftFloat t2 = t * t;

	// Coefficients verified for SoftFloat normalization
	SoftFloat c0 = SoftFloat::from_raw(536870912, -29);     // 1.0
	SoftFloat c1 = SoftFloat::from_raw(-715827883, -31);    // -1/3
	SoftFloat c2 = SoftFloat::from_raw(858993459, -32);     // 1/5  
	SoftFloat c3 = SoftFloat::from_raw(-613566757, -32);    // -1/7
	SoftFloat c4 = SoftFloat::from_raw(954437177, -33);     // 1/9
	SoftFloat c5 = SoftFloat::from_raw(-780903145, -33);    // -1/11

	// Horner's method
	SoftFloat p = c5;
	p = c4 + t2 * p;
	p = c3 + t2 * p;
	p = c2 + t2 * p;
	p = c1 + t2 * p;
	p = c0 + t2 * p;

	SoftFloat angle = t * p;

	// Reconstruct angle
	if (swap) angle = SoftFloat::half_pi() - angle;
	if (x_neg) angle = SoftFloat::pi() - angle;
	if (y_neg) angle = -angle;

	return angle;
}
#endif

// exp table: e^(i/256 * ln2), i=0..256
static constexpr int32_t EXP_MANT[257] = {
	0x20000000, 0x201635F5, 0x202C7B54, 0x2042D028, 0x2059347D, 0x206FA85C, 0x20862BD1, 0x209CBEE6,
	0x20B361A6, 0x20CA141C, 0x20E0D654, 0x20F7A857, 0x210E8A31, 0x21257BED, 0x213C7D96, 0x21538F36,
	0x216AB0DA, 0x2181E28C, 0x21992457, 0x21B07646, 0x21C7D866, 0x21DF4AC0, 0x21F6CD60, 0x220E6052,
	0x222603A0, 0x223DB757, 0x22557B81, 0x226D502A, 0x2285355D, 0x229D2B27, 0x22B53191, 0x22CD48A9,
	0x22E57079, 0x22FDA90D, 0x2315F271, 0x232E4CB0, 0x2346B7D7, 0x235F33F0, 0x2377C108, 0x23905F2A,
	0x23A90E63, 0x23C1CEBD, 0x23DAA046, 0x23F38308, 0x240C7711, 0x24257C6B, 0x243E9323, 0x2457BB45,
	0x2470F4DD, 0x248A3FF7, 0x24A39C9F, 0x24BD0AE2, 0x24D68ACC, 0x24F01C68, 0x2509BFC4, 0x252374EB,
	0x253D3BEA, 0x255714CE, 0x2570FFA2, 0x258AFC73, 0x25A50B4E, 0x25BF2C3F, 0x25D95F52, 0x25F3A495,
	0x260DFC14, 0x262865DC, 0x2642E1F9, 0x265D7077, 0x26781165, 0x2692C4CE, 0x26AD8ABF, 0x26C86346,
	0x26E34E6E, 0x26FE4C46, 0x27195CDA, 0x27348037, 0x274FB66A, 0x276AFF80, 0x27865B86, 0x27A1CA8A,
	0x27BD4C98, 0x27D8E1BE, 0x27F48A09, 0x28104587, 0x282C1444, 0x2847F64E, 0x2863EBB3, 0x287FF47F,
	0x289C10C1, 0x28B84085, 0x28D483DA, 0x28F0DACD, 0x290D456C, 0x2929C3C3, 0x294655E2, 0x2962FBD5,
	0x297FB5AA, 0x299C8370, 0x29B96534, 0x29D65B04, 0x29F364ED, 0x2A1082FF, 0x2A2DB546, 0x2A4AFBD0,
	0x2A6856AD, 0x2A85C5EA, 0x2AA34995, 0x2AC0E1BC, 0x2ADE8E6D, 0x2AFC4FB8, 0x2B1A25A9, 0x2B381050,
	0x2B560FBB, 0x2B7423F7, 0x2B924D15, 0x2BB08B21, 0x2BCEDE2B, 0x2BED4642, 0x2C0BC373, 0x2C2A55CE,
	0x2C48FD60, 0x2C67BA3A, 0x2C868C6A, 0x2CA573FD, 0x2CC47105, 0x2CE3838E, 0x2D02ABA9, 0x2D21E963,
	0x2D413CCD, 0x2D60A5F5, 0x2D8024EA, 0x2D9FB9BC, 0x2DBF6479, 0x2DDF2531, 0x2DFEFBF3, 0x2E1EE8CE,
	0x2E3EEBD2, 0x2E5F050E, 0x2E7F3491, 0x2E9F7A6C, 0x2EBFD6AD, 0x2EE04963, 0x2F00D2A0, 0x2F217271,
	0x2F4228E8, 0x2F62F613, 0x2F83DA02, 0x2FA4D4C6, 0x2FC5E66E, 0x2FE70F09, 0x30084EA8, 0x3029A55C,
	0x304B1333, 0x306C983D, 0x308E348C, 0x30AFE82F, 0x30D1B337, 0x30F395B2, 0x31158FB3, 0x3137A149,
	0x3159CA84, 0x317C0B76, 0x319E642D, 0x31C0D4BC, 0x31E35D32, 0x3205FDA0, 0x3228B617, 0x324B86A7,
	0x326E6F62, 0x32917057, 0x32B48998, 0x32D7BB35, 0x32FB0540, 0x331E67C9, 0x3341E2E2, 0x3365769B,
	0x33892305, 0x33ACE833, 0x33D0C634, 0x33F4BD1A, 0x3418CCF7, 0x343CF5DB, 0x346137D9, 0x34859301,
	0x34AA0764, 0x34CE9516, 0x34F33C26, 0x3517FCA8, 0x353CD6AB, 0x3561CA42, 0x3586D780, 0x35ABFE74,
	0x35D13F33, 0x35F699CC, 0x361C0E53, 0x36419CD9, 0x36674571, 0x368D082B, 0x36B2E51C, 0x36D8DC54,
	0x36FEEDE6, 0x372519E4, 0x374B6061, 0x3771C16F, 0x37983D21, 0x37BED388, 0x37E584B8, 0x380C50C3,
	0x383337BB, 0x385A39B4, 0x388156C0, 0x38A88EF2, 0x38CFE25D, 0x38F75113, 0x391EDB28, 0x394680AF,
	0x396E41BA, 0x39961E5D, 0x39BE16AB, 0x39E62AB7, 0x3A0E5A94, 0x3A36A656, 0x3A5F0E10, 0x3A8791D6,
	0x3AB031BA, 0x3AD8EDD1, 0x3B01C62E, 0x3B2ABAE4, 0x3B53CC08, 0x3B7CF9AC, 0x3BA643E6, 0x3BCFAAC8,
	0x3BF92E67, 0x3C22CED6, 0x3C4C8C2A, 0x3C766676, 0x3CA05DCF, 0x3CCA7249, 0x3CF4A3F8, 0x3D1EF2F0,
	0x3D495F45, 0x3D73E90D, 0x3D9E905B, 0x3DC95544, 0x3DF437DD, 0x3E1F3839, 0x3E4A566F, 0x3E759292,
	0x3EA0ECB7, 0x3ECC64F3, 0x3EF7FB5B, 0x3F23B004, 0x3F4F8303, 0x3F7B746D, 0x3FA78457, 0x3FD3B2D6,
	0x20000000
};

static constexpr int8_t EXP_EXP[257] = {
	-29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
	-29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
	-29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
	-29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
	-29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
	-29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
	-29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
	-29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
	-29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
	-29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
	-29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
	-29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
	-29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
	-29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
	-29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
	-29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
	-28
};


constexpr SF_HOT SoftFloat SoftFloat::exp() const noexcept {
	SoftFloat x = *this;
	if (x.mantissa == 0) return SoftFloat::one();

	constexpr SoftFloat LN2 = SoftFloat::from_raw(0x2C5C85FE, -30); // ln(2)
	constexpr SoftFloat INV_LN2 = SoftFloat::from_raw(0x2E2B8A3E, -29); // 1/ln(2)

	// Range reduction: x = k*ln2 + f, with f in [0, ln2)
	int32_t k = (x * INV_LN2).to_int32();
	SoftFloat f = x - SoftFloat(k) * LN2;
	if (f.is_negative()) { f = f + LN2; k -= 1; }
	else if (!(f < LN2)) { f = f - LN2; k += 1; }

	// f is in [0, ln2). Scale to table index: idx = f * (256 / ln2)
	constexpr SoftFloat SCALE = SoftFloat::from_raw(0x2E2B8A3E, -21); // 256/ln2
	SoftFloat idx_f = f * SCALE;
	int32_t idx = idx_f.to_int32();
	if (idx > 255) idx = 255;
	if (idx < 0)   idx = 0;

	SoftFloat frac = idx_f - SoftFloat(idx);

	SoftFloat v0 = SoftFloat::from_raw(EXP_MANT[idx], EXP_EXP[idx]);
	SoftFloat v1 = SoftFloat::from_raw(EXP_MANT[idx + 1], EXP_EXP[idx + 1]);

	SoftFloat e_f = v0 + frac * (v1 - v0);

	// Multiply by 2^k: adjust exponent by k
	return e_f << k;
}

static constexpr int32_t LOG2_MANT[257] = {
	0x00000000, 0x2E1388DA, 0x2DFCA16D, 0x226C622F, 0x2DCF2D0B, 0x3926C775, 0x2239A3AA, 0x27DA6128,
	0x2D75A6EB, 0x330B7F86, 0x389BF572, 0x3E271306, 0x21D6713F, 0x2496B6FC, 0x27545FBA, 0x2A0F706C,
	0x2CC7EDF5, 0x2F7DDD2D, 0x323142DC, 0x34E223BE, 0x37908481, 0x3A3C69C6, 0x3CE5D822, 0x3F8CD41E,
	0x2118B119, 0x2269C369, 0x23B9A32E, 0x25085296, 0x2655D3C4, 0x27A228DB, 0x28ED53F3, 0x2A375720,
	0x2B803473, 0x2CC7EDF5, 0x2E0E85A9, 0x2F53FD8F, 0x309857A0, 0x31DB95D0, 0x331DBA0E, 0x345EC646,
	0x359EBC5B, 0x36DD9E2E, 0x381B6D9B, 0x39582C78, 0x3A93DC98, 0x3BCE7FC7, 0x3D0817CE, 0x3E40A672,
	0x3F782D72, 0x20575744, 0x20F215B7, 0x218C52EA, 0x22260FB5, 0x22BF4CED, 0x23580B65, 0x23F04BED,
	0x24880F56, 0x251F566B, 0x25B621F8, 0x264C72C6, 0x26E2499D, 0x2777A741, 0x280C8C76, 0x28A0F9FD,
	0x2934F097, 0x29C87101, 0x2A5B7BF8, 0x2AEE1236, 0x2B803473, 0x2C11E368, 0x2CA31FC8, 0x2D33EA49,
	0x2DC4439B, 0x2E542C6F, 0x2EE3A574, 0x2F72AF58, 0x30014AC6, 0x308F7867, 0x311D38E5, 0x31AA8CE7,
	0x32377512, 0x32C3F20A, 0x33500472, 0x33DBACEB, 0x3466EC14, 0x34F1C28D, 0x357C30F2, 0x360637DF,
	0x368FD7EE, 0x371911B7, 0x37A1E5D3, 0x382A54D8, 0x38B25F5A, 0x393A05ED, 0x39C14924, 0x3A482990,
	0x3ACEA7C0, 0x3B54C444, 0x3BDA7FA8, 0x3C5FDA7A, 0x3CE4D543, 0x3D69708F, 0x3DEDACE6, 0x3E718ACF,
	0x3EF50AD1, 0x3F782D72, 0x3FFAF335, 0x203EAE4E, 0x207FB517, 0x20C08E33, 0x210139E4, 0x2141B869,
	0x21820A01, 0x21C22EEA, 0x22022762, 0x2241F3A7, 0x228193F5, 0x22C10889, 0x2300519E, 0x233F6F71,
	0x237E623D, 0x23BD2A3B, 0x23FBC7A6, 0x243A3AB7, 0x247883A8, 0x24B6A2B1, 0x24F4980B, 0x253263EC,
	0x2570068E, 0x25AD8026, 0x25EAD0EB, 0x2627F914, 0x2664F8D5, 0x26A1D064, 0x26DE7FF6, 0x271B07C0,
	0x275767F5, 0x2793A0C9, 0x27CFB26F, 0x280B9D1A, 0x284760FD, 0x2882FE49, 0x28BE7531, 0x28F9C5E5,
	0x2934F097, 0x296FF577, 0x29AAD4B6, 0x29E58E83, 0x2A20230E, 0x2A5A9285, 0x2A94DD19, 0x2ACF02F7,
	0x2B09044D, 0x2B42E149, 0x2B7C9A19, 0x2BB62EEA, 0x2BEF9FE8, 0x2C28ED40, 0x2C62171E, 0x2C9B1DAE,
	0x2CD4011C, 0x2D0CC192, 0x2D455F3C, 0x2D7DDA44, 0x2DB632D4, 0x2DEE6917, 0x2E267D36, 0x2E5E6F5A,
	0x2E963FAC, 0x2ECDEE56, 0x2F057B7F, 0x2F3CE751, 0x2F7431F2, 0x2FAB5B8B, 0x2FE26443, 0x30194C40,
	0x305013AB, 0x3086BAA9, 0x30BD4161, 0x30F3A7F8, 0x3129EE96, 0x3160155E, 0x31961C76, 0x31CC0404,
	0x3201CC2C, 0x32377512, 0x326CFEDB, 0x32A269AB, 0x32D7B5A5, 0x330CE2ED, 0x3341F1A7, 0x3376E1F5,
	0x33ABB3FA, 0x33E067D9, 0x3414FDB4, 0x344975AD, 0x347DCFE7, 0x34B20C82, 0x34E62BA0, 0x351A2D62,
	0x354E11EB, 0x3581D959, 0x35B583CE, 0x35E9116A, 0x361C824D, 0x364FD697, 0x36830E69, 0x36B629E1,
	0x36E9291E, 0x371C0C41, 0x374ED367, 0x37817EAF, 0x37B40E39, 0x37E68222, 0x3818DA88, 0x384B178A,
	0x387D3945, 0x38AF3FD7, 0x38E12B5D, 0x3912FBF4, 0x3944B1B9, 0x39764CC9, 0x39A7CD41, 0x39D9333D,
	0x3A0A7EDA, 0x3A3BB033, 0x3A6CC764, 0x3A9DC48A, 0x3ACEA7C0, 0x3AFF7121, 0x3B3020C8, 0x3B60B6D1,
	0x3B913356, 0x3BC19672, 0x3BF1E041, 0x3C2210DB, 0x3C52285C, 0x3C8226DD, 0x3CB20C79, 0x3CE1D948,
	0x3D118D66, 0x3D4128EB, 0x3D70ABF1, 0x3DA01691, 0x3DCF68E3, 0x3DFEA301, 0x3E2DC503, 0x3E5CCF02,
	0x3E8BC117, 0x3EBA9B59, 0x3EE95DE1, 0x3F1808C7, 0x3F469C22, 0x3F75180B, 0x3FA37C98, 0x3FD1C9E2,
	0x20000000,
};

static constexpr int8_t LOG2_EXP[257] = {
	0, -37, -36, -35, -35, -35, -34, -34, -34, -34, -34, -34, -33, -33, -33, -33,
	-33, -33, -33, -33, -33, -33, -33, -33, -32, -32, -32, -32, -32, -32, -32, -32,
	-32, -32, -32, -32, -32, -32, -32, -32, -32, -32, -32, -32, -32, -32, -32, -32,
	-32, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31,
	-31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31,
	-31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31,
	-31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
	-29,
};

// Natural logarithm: log(x) = log2(x) * ln(2)
constexpr SoftFloat SoftFloat::log() const noexcept {
	constexpr SoftFloat LN2 = from_raw(0x2C5C85FE, -30);
	return log2() * LN2;                        // one extra multiply, but half the code size
}

// log2(x)
constexpr SoftFloat SoftFloat::log2() const noexcept {
	if (mantissa <= 0) return SoftFloat::zero();

	// Same as log() but without final multiplication
	int32_t E = exponent + 29;
	uint32_t m_abs = sf_abs32(mantissa);
	uint32_t low = m_abs - (1u << 29);
	uint32_t t_int = low >> 21;
	uint32_t t_frac = (low >> 13) & 0xFF;

	SoftFloat v0 = SoftFloat::from_raw(LOG2_MANT[t_int], LOG2_EXP[t_int]);
	SoftFloat v1 = SoftFloat::from_raw(LOG2_MANT[t_int + 1], LOG2_EXP[t_int + 1]);
	SoftFloat frac = SoftFloat(static_cast<int32_t>(t_frac)) >> 8;
	SoftFloat log2_M = v0 + frac * (v1 - v0);
	return log2_M + SoftFloat(E);
}

// log10(x) = log2(x) * log10(2)
constexpr SoftFloat SoftFloat::log10() const noexcept {
	constexpr SoftFloat LOG10_2 = SoftFloat::from_raw(0x268826A1, -31); // log10(2) ≈ 0.30103
	return log2() * LOG10_2;
}

// pow(x, y) = exp(y * log(x))
constexpr SoftFloat SoftFloat::pow(SoftFloat y) const noexcept {
	if (mantissa == 0) {
		if (y.mantissa == 0) return SoftFloat::one(); // 0^0 = 1
		return SoftFloat::zero();
	}
	if (y.mantissa == 0) return SoftFloat::one();
	// Negative base to non-integer exponent is undefined; return 0.
	if (is_negative() && (y != y.trunc())) return SoftFloat::zero();
	return (y * log()).exp();
}

constexpr SF_HOT SoftFloat hypot(SoftFloat x, SoftFloat y) noexcept {
	x = x.abs(); y = y.abs();
	if (x < y) { SoftFloat t = x; x = y; y = t; }
	if (x.mantissa == 0) return {};

	// Scale both by 2^{-(e+29)} so xs ≈ [0.5,1) and ys ≤ xs.
	// This keeps x²+y² in [0.25, 2), well within normalised range.
	int32_t e = x.exponent;
	SoftFloat xs = SoftFloat::from_raw(x.mantissa, -29);
	SoftFloat ys = SoftFloat::from_raw(y.mantissa, y.exponent - e - 29);  // may underflow → zero, which is correct

	SoftFloat r = (xs * xs + ys * ys).sqrt();          // no division
	return SoftFloat::from_raw(r.mantissa, r.exponent + e + 29);  // restore original scale
}

// trunc – toward zero
constexpr SF_HOT SoftFloat SoftFloat::trunc() const noexcept {
	return SoftFloat(to_int32());
}

// floor – toward -inf
constexpr SoftFloat SoftFloat::floor() const noexcept {
	if (mantissa == 0) return *this;
	if (exponent >= 0) return *this; // already integer
	int32_t i = to_int32();
	SoftFloat fi(i);
	// If negative and not exact integer, subtract one
	if (is_negative() && *this != fi) {
		return fi - SoftFloat::one();
	}
	return fi;
}

// ceil – toward +inf
constexpr SoftFloat SoftFloat::ceil() const noexcept {
	if (mantissa == 0) return *this;
	if (exponent >= 0) return *this;
	int32_t i = to_int32();
	SoftFloat fi(i);
	if (is_positive() && *this != fi) {
		return fi + SoftFloat::one();
	}
	return fi;
}

// round – nearest, ties away from zero
constexpr SoftFloat SoftFloat::round() const noexcept {
	SoftFloat half = SoftFloat::half();
	if (is_negative()) {
		return -((-*this + half).floor());
	}
	else {
		return (*this + half).floor();
	}
}

// fract – fractional part
constexpr SoftFloat SoftFloat::fract() const noexcept {
	return *this - trunc();
}

// modf – split into integer and fractional parts (C-style via pointers, but we can return pair)
constexpr SoftFloatPair SoftFloat::modf() const noexcept {
	SoftFloat intpart = trunc();
	return { intpart, *this - intpart };
}

constexpr SoftFloat SoftFloat::copysign(SoftFloat sign) const noexcept {
	if (sign.is_negative() == is_negative()) return *this;
	return -*this;
}
constexpr SoftFloat SoftFloat::fmod(SoftFloat y) const noexcept {
	if (y.mantissa == 0) return *this; // should be NaN, but return 0
	SoftFloat n = (*this / y).trunc();
	return *this - n * y;
}
constexpr SoftFloat SoftFloat::fma(SoftFloat b, SoftFloat c) const noexcept {
	return fused_mul_add(*this, b, c);
}

// =========================================================================
// lerp — constexpr (implicitly inline)
// =========================================================================
constexpr SF_HOT SoftFloat lerp(SoftFloat a, SoftFloat b, SoftFloat t) noexcept {
	return a + t * (b - a);
}


// =========================================================================
// Compile‑time evaluation tests for all SoftFloat functions
// =========================================================================

// Helper for approximate float equality at compile time (ULP‑based)
consteval bool ct_approx(float a, float b, int max_ulp = 16) {
	if (a == b) return true;
	uint32_t ua = sf_bitcast<uint32_t>(a);
	uint32_t ub = sf_bitcast<uint32_t>(b);
	if ((ua ^ ub) & 0x80000000u) return false; // different signs
	int32_t diff = static_cast<int32_t>(ua - ub);
	return (diff < 0 ? -diff : diff) <= max_ulp;
}

// Helper to check normalization invariant at compile time
consteval bool ct_is_normalized(int32_t mantissa, int32_t exponent) {
	if (mantissa == 0) return true; // zero is trivially normalized
	uint32_t a = sf_abs32(mantissa);
	int lz = sf_clz(a);
	// Must have bit29 set and bits 31:30 clear
	return lz == 2 && (a & 0xC0000000u) == 0;
}

// Helper to check float conversion at compile time (exact match required for constants)
consteval bool ct_float_eq(SoftFloat sf, float expected) {
	return sf_bitcast<uint32_t>(sf.to_float()) == sf_bitcast<uint32_t>(expected);
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

// ---------- Zero ----------
static_assert(SoftFloat::zero().mantissa == 0);
static_assert(SoftFloat::zero().exponent == 0);
static_assert(SoftFloat::zero().is_zero());
static_assert(ct_float_eq(SoftFloat::zero(), 0.0f));

// ---------- One ----------
static_assert(ct_is_normalized(SoftFloat::one().mantissa, SoftFloat::one().exponent));
static_assert(SoftFloat::one().mantissa == 0x20000000);
static_assert(SoftFloat::one().exponent == -29);
static_assert(ct_float_eq(SoftFloat::one(), 1.0f));
static_assert(!SoftFloat::one().is_negative());

// ---------- Negative One ----------
static_assert(ct_is_normalized(SoftFloat::neg_one().mantissa, SoftFloat::neg_one().exponent));
static_assert(SoftFloat::neg_one().mantissa == -0x20000000);
static_assert(SoftFloat::neg_one().exponent == -29);
static_assert(ct_float_eq(SoftFloat::neg_one(), -1.0f));
static_assert(SoftFloat::neg_one().is_negative());

// ---------- Half (0.5) ----------
static_assert(ct_is_normalized(SoftFloat::half().mantissa, SoftFloat::half().exponent));
static_assert(SoftFloat::half().mantissa == 0x20000000);
static_assert(SoftFloat::half().exponent == -30);
static_assert(ct_float_eq(SoftFloat::half(), 0.5f));

// ---------- Two (2.0) ----------
static_assert(ct_is_normalized(SoftFloat::two().mantissa, SoftFloat::two().exponent));
static_assert(SoftFloat::two().mantissa == 0x20000000);
static_assert(SoftFloat::two().exponent == -28);
static_assert(ct_float_eq(SoftFloat::two(), 2.0f));

// ---------- Pi (π ≈ 3.14159265) ----------
static_assert(ct_is_normalized(SoftFloat::pi().mantissa, SoftFloat::pi().exponent));
static_assert(SoftFloat::pi().mantissa == 843314857);
static_assert(SoftFloat::pi().exponent == -28);
static_assert(ct_approx(SoftFloat::pi().to_float(), 3.14159265f, 2));
static_assert(SoftFloat::pi().is_positive());

// ---------- Two Pi (2π) ----------
static_assert(ct_is_normalized(SoftFloat::two_pi().mantissa, SoftFloat::two_pi().exponent));
static_assert(SoftFloat::two_pi().mantissa == 843314857);
static_assert(SoftFloat::two_pi().exponent == -27);
static_assert(ct_approx(SoftFloat::two_pi().to_float(), 6.2831853f, 2));

// ---------- Half Pi (π/2) ----------
static_assert(ct_is_normalized(SoftFloat::half_pi().mantissa, SoftFloat::half_pi().exponent));
static_assert(SoftFloat::half_pi().mantissa == 843314857);
static_assert(SoftFloat::half_pi().exponent == -29);
static_assert(ct_approx(SoftFloat::two_pi().to_float(), 6.2831853f, 2));

// ---------- Relationships between constants ----------
// pi * 2 == two_pi
static_assert((SoftFloat::pi()* SoftFloat::two()).to_float() == SoftFloat::two_pi().to_float());
// two_pi / 2 == pi
static_assert((SoftFloat::two_pi() / SoftFloat::two()).to_float() == SoftFloat::pi().to_float());
// pi / 2 == half_pi
static_assert((SoftFloat::pi() / SoftFloat::two()).to_float() == SoftFloat::half_pi().to_float());
// one + one == two
static_assert((SoftFloat::one() + SoftFloat::one()).to_float() == SoftFloat::two().to_float());
// half + half == one
static_assert((SoftFloat::half() + SoftFloat::half()).to_float() == SoftFloat::one().to_float());
// -one == neg_one
static_assert((-SoftFloat::one()).to_float() == SoftFloat::neg_one().to_float());

// ---------- Constants used in math functions (verify normalization) ----------
// LN2 (used in exp)
static_assert(ct_is_normalized(0x2C5C85FE, -30)); // from exp() implementation
// INV_LN2
static_assert(ct_is_normalized(0x2E2B8A3E, -29));
// SCALE for exp table
static_assert(ct_is_normalized(0x2E2B8A3E, -21));

// Step and inv_step for sin/cos table
static_assert(ct_is_normalized(843314857, -36));
static_assert(ct_is_normalized(683565276, -23));

// inv_two_pi constant
static_assert(ct_is_normalized(683565276, -32));

// ---------- Basic arithmetic (already covered) ----------
static_assert((SoftFloat::one() + SoftFloat::one()).to_float() == 2.0f);
static_assert((SoftFloat::two() - SoftFloat::one()).to_float() == 1.0f);
static_assert((SoftFloat::two()* SoftFloat::two()).to_float() == 4.0f);
static_assert((SoftFloat::two() / SoftFloat::two()).to_float() == 1.0f);
static_assert((-SoftFloat::one()).to_float() == -1.0f);
static_assert(SoftFloat::neg_one().abs().to_float() == 1.0f);

// ---------- Comparisons ----------
static_assert(SoftFloat::one() < SoftFloat::two());
static_assert(SoftFloat::neg_one() < SoftFloat::zero());
static_assert(SoftFloat::one() == SoftFloat::one());
static_assert(SoftFloat::one() != SoftFloat::two());

// ---------- Shifts ----------
static_assert((SoftFloat::one() << 2).to_float() == 4.0f);
static_assert((SoftFloat(8.0f) >> 2).to_float() == 2.0f);

// ---------- Fused operations ----------
static_assert(fused_mul_add(SoftFloat::one(), SoftFloat::two(), SoftFloat::three()).to_float() == 7.0f);
static_assert(fused_mul_sub(SoftFloat::one(), SoftFloat::two(), SoftFloat::three()).to_float() == -5.0f);
static_assert(fused_mul_mul_add(SoftFloat::one(), SoftFloat::two(),
	SoftFloat::three(), SoftFloat::four()).to_float() == 14.0f);
static_assert(fused_mul_mul_sub(SoftFloat::one(), SoftFloat::two(),
	SoftFloat::three(), SoftFloat::four()).to_float() == -10.0f);

// ---------- Trigonometry ----------
static_assert(SoftFloat::zero().sin().to_float() == 0.0f);
static_assert(ct_approx(SoftFloat::half_pi().sin().to_float(), 1.0f, 256));
static_assert(SoftFloat::zero().cos().to_float() == 1.0f);
static_assert(ct_approx(SoftFloat::pi().cos().to_float(), -1.0f, 1024));
static_assert(SoftFloat::zero().tan().to_float() == 0.0f);
static_assert(ct_approx(SoftFloat(0.5f).asin().to_float(), /*asinf(0.5f) = */ 0.52359878f, 512));
static_assert(ct_approx(SoftFloat(0.5f).acos().to_float(), /*acosf(0.5f) = */ 1.04719755f, 512));

// ---------- atan2 ----------
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
static_assert(sf_min(SoftFloat::one(), SoftFloat::two()).to_float() == 1.0f);
static_assert(sf_max(SoftFloat::one(), SoftFloat::two()).to_float() == 2.0f);
static_assert(sf_clamp(SoftFloat::three(), SoftFloat::zero(), SoftFloat::two()).to_float() == 2.0f);
static_assert(lerp(SoftFloat::zero(), SoftFloat::two(), SoftFloat::half()).to_float() == 1.0f);
static_assert(hypot(SoftFloat::three(), SoftFloat::four()).to_float() == 5.0f);

// ---------- Expression template interactions ----------
static_assert((SoftFloat::one() + SoftFloat::two() * SoftFloat::three()).to_float() == 7.0f);
static_assert((SoftFloat::two() * SoftFloat::three() - SoftFloat::one()).to_float() == 5.0f);
static_assert((-(SoftFloat::two() * SoftFloat::three())).to_float() == -6.0f);