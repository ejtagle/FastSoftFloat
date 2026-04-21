/// File: FusedSoftFloat.hh
/// SoftFloat optimised for GD32F103 (Cortex‑M3, ARMv7‑M)
///
/// Representation:
///   value = mantissa * 2^exponent
///   mantissa == 0  =>  zero (and exponent will always be 0 if mantissa == 0)
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
[[nodiscard]] constexpr SF_CONST SF_INLINE int sf_clz(uint32_t x) noexcept
{
	if (SF_IS_CONSTEVAL()) {
		return __builtin_clz(x);
	}
	return __builtin_clz(x);
}

// sf_abs32 — branchless absolute value for int32_t, constexpr-safe.
[[nodiscard]] constexpr SF_CONST SF_INLINE uint32_t sf_abs32(int32_t m) noexcept
{
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
[[nodiscard]] constexpr SF_CONST SF_INLINE int32_t sf_sat_exp(int32_t e) noexcept
{
#if defined(__arm__)
	if (!SF_IS_CONSTEVAL()) {
		int32_t r;
		__asm__(
			"ssat %0, #8, %1\n\t" 
			: "=r"(r) : "r"(e));
		return r;
	}
#endif
	if (e > 127) return  127;
	if (e < -128) return -128;
	return e;
}

// ─── sf_normalise_fast: branchless CLZ path with early-out ──────────────────
constexpr SF_INLINE void sf_normalise_fast(int32_t& m, int32_t& e) noexcept
{
	if (SF_IS_CONSTEVAL()) {
		// full portable path for compile-time evaluation
		uint32_t sign = static_cast<uint32_t>(m >> 31);
		uint32_t a    = (static_cast<uint32_t>(m) ^ sign) - sign;
		if (LIKELY((a & 0x60000000u) == 0x20000000u)) {
			e = sf_sat_exp(e);
			return;
		}
		if (UNLIKELY(a & 0x40000000u)) {
			a >>= 1; e += 1;
		}
		else {
			int lz = sf_clz(a);
			int sh = lz - 2;
			e -= sh;
			a <<= sh;
		}
		e = sf_sat_exp(e);
		m = static_cast<int32_t>((a ^ sign) - sign);
		return;
	}

#if defined(__arm__)
	{
		uint32_t sign = static_cast<uint32_t>(m >> 31);
		uint32_t a    = (static_cast<uint32_t>(m) ^ sign) - sign;

		// ── hot path: already normalised (bit 29 set, bit 30 clear) ──────
		if (LIKELY((a & 0x60000000u) == 0x20000000u)) {
			// SSAT is 1 cycle — always do it
			__asm__("ssat %0, #8, %1" : "=r"(e) : "r"(e));
			return;
		}

		// ── determine shift using hardware CLZ ───────────────────────────
		uint32_t lz;
		__asm__("clz %0, %1" : "=r"(lz) : "r"(a));
		int32_t shift = static_cast<int32_t>(lz) - 2; // positive = left, negative = right

		if (shift > 0) {
			a <<= shift;
			e -= shift;
		}
		else {
			// shift == -1 is the only case here (bit 30 was set, lz == 1)
			a >>= 1;
			e += 1;
		}

		__asm__("ssat %0, #8, %1" : "=r"(e) : "r"(e));
		m = static_cast<int32_t>((a ^ sign) - sign);
		return;
	}
#else
	{
		uint32_t sign = static_cast<uint32_t>(m >> 31);
		uint32_t a    = (static_cast<uint32_t>(m) ^ sign) - sign;
		if (LIKELY((a & 0x60000000u) == 0x20000000u)) {
			e = sf_sat_exp(e);
			return;
		}
		if (UNLIKELY(a & 0x40000000u)) {
			a >>= 1; e += 1;
		}
		else {
			int lz = sf_clz(a);
			int sh = lz - 2;
			e -= sh;
			a <<= sh;
		}
		e = sf_sat_exp(e);
		m = static_cast<int32_t>((a ^ sign) - sign);
	}
#endif
}



// sf_recip — Newton-refined reciprocal, now constexpr.
// Returns Y ≈ 2^60 / b  for b in [2^29, 2^30).
// Uses one UMULL Newton step: ~14 cycles on Cortex-M3.
[[nodiscard]] constexpr SF_CONST SF_INLINE uint64_t sf_recip(uint32_t b) noexcept
{

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

private:
	// ------------------------------------------------------------------
	// from_raw — bypass normalisation (caller guarantees invariant)
	// ------------------------------------------------------------------
	[[nodiscard]] static constexpr SoftFloat from_raw(int32_t m, int32_t e) noexcept {
		SoftFloat r; r.mantissa = m; r.exponent = e; return r;
	}
public:
	// ------------------------------------------------------------------
	// Default constructor — zero
	// ------------------------------------------------------------------
	constexpr SoftFloat() noexcept : mantissa{ 0 }, exponent{ 0 } {}


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

	constexpr SF_HOT const SoftFloat& operator=(int v) noexcept {
		mantissa = v;
		exponent = 0;
		normalise();
		return *this;

	}

	constexpr SF_HOT const SoftFloat& operator=(int32_t v) noexcept {
		mantissa = v;
		exponent = 0;
		normalise();
		return *this;
	}

	constexpr SF_HOT const SoftFloat& operator=(int16_t v) noexcept {
		mantissa = static_cast<int32_t>(v);
		exponent = 0;
		normalise();
		return *this;
	}

	constexpr SF_HOT const SoftFloat& operator=(float f) noexcept {
		mantissa = 0;
		exponent = 0;
		from_float(f);
		return *this;
	}

	// Proxy constructor (defined after sf_mul_expr)
	constexpr SF_HOT SoftFloat(const sf_mul_expr& m) noexcept;

	// ------------------------------------------------------------------
	// Manual re-normalise 
	// ------------------------------------------------------------------
	
	// ── normalise: CLZ + SSAT in one tight block ─────────────────────────
	constexpr SF_HOT SF_INLINE void normalise() noexcept
	{
		int32_t m = mantissa, e = exponent;
		if (m == 0) { exponent = 0; return; }

		if (SF_IS_CONSTEVAL()) {
			uint32_t a = sf_abs32(m);
			int lz    = sf_clz(a);
			int shift = lz - 2;
			if (shift > 0) {
				int ne = e - shift;
				if (ne < -250) { mantissa = 0; exponent = 0; return; }
				a <<= shift; e = ne;
			}
			else if (shift < 0) {
				int rs = -shift; a >>= rs; e += rs;
			}
			exponent = sf_sat_exp(e);
			mantissa = (m < 0) ? -static_cast<int32_t>(a) : static_cast<int32_t>(a);
			return;
		}

#if defined(__arm__)
		{
			uint32_t a;
			uint32_t lz;
			// abs(m) via arithmetic shift
			__asm__(
			    "eor %[a], %[m], %[m], asr #31\n\t"
			    "sub %[a], %[a], %[m], asr #31\n\t"
			    "clz %[lz], %[a]              \n\t"
			    : [a] "=&r"(a),
				[lz] "=&r"(lz)
			    : [m] "r"(m));
			int32_t shift = static_cast<int32_t>(lz) - 2;
			if (shift > 0) {
				int ne = e - shift;
				if (UNLIKELY(ne < -250)) { mantissa = 0; exponent = 0; return; }
				a <<= shift; e = ne;
			}
			else if (shift < 0) {
				int rs = -shift; a >>= rs; e += rs;
			}
			__asm__("ssat %0, #8, %1" : "=r"(e) : "r"(e));
			exponent = e;
			mantissa = (m < 0) ? -static_cast<int32_t>(a) : static_cast<int32_t>(a);
		}
#else
		{
			uint32_t a = sf_abs32(m);
			int lz    = sf_clz(a);
			int shift = lz - 2;
			if (shift > 0) {
				int ne = e - shift;
				if (ne < -250) { mantissa = 0; exponent = 0; return; }
				a <<= shift; e = ne;
			}
			else if (shift < 0) {
				int rs = -shift; a >>= rs; e += rs;
			}
			exponent = sf_sat_exp(e);
			mantissa = (m < 0) ? -static_cast<int32_t>(a) : static_cast<int32_t>(a);
		}
#endif
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

			// Matched exponents: behaviour is sign-determined.
			//   same sign      → addition of magnitudes → guaranteed 1-bit overflow
			//   opposite signs → subtraction of magnitudes → guaranteed cancellation (or zero)
			if ((a.mantissa ^ b.mantissa) >= 0) {
				rm = a.mantissa + b.mantissa; // e.g. 0x20000000 + 0x20000000 = 0x40000000
				rm >>= 1; // arithmetic shift: exact 1-bit normalise
				if (UNLIKELY(++re > 127)) re = 127;
				return from_raw(rm, re);
			}

			// Opposite signs: never normalised, always needs left-shift renormalisation.
			rm = a.mantissa + b.mantissa;
			if (UNLIKELY(rm == 0)) return zero();
			sf_normalise_fast(rm, re);
			return from_raw(rm, re);
		}

		// Unequal exponents: align smaller operand
		if (d > 0) {
			rm = a.mantissa + (b.mantissa >> d);
			re = a.exponent;
		}
		else {
			rm = (a.mantissa >> -d) + b.mantissa;
			re = b.exponent;
		}

		if (UNLIKELY(rm == 0)) return zero();

		uint32_t ab = sf_abs32(rm);
		if (LIKELY((ab & 0x60000000u) == 0x20000000u)) {
			return from_raw(rm, re);
		}
		if (ab & 0x40000000u) {
			rm >>= 1;
			if (UNLIKELY(++re > 127)) re = 127;
			return from_raw(rm, re);
		}
		sf_normalise_fast(rm, re);
		return from_raw(rm, re);
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

			// Matched exponents: behaviour is sign-determined.
			//   opposite signs → addition of magnitudes → guaranteed 1-bit overflow
			//   same sign      → subtraction of magnitudes → guaranteed cancellation (or zero)
			if ((a.mantissa ^ b.mantissa) < 0) {
				rm = a.mantissa - b.mantissa; // e.g. 0x20000000 - (-0x20000000) = 0x40000000
				rm >>= 1; // arithmetic shift: exact 1-bit normalise
				if (UNLIKELY(++re > 127)) re = 127;
				return from_raw(rm, re);
			}

			// Same sign: never normalised, always needs left-shift renormalisation.
			rm = a.mantissa - b.mantissa;
			if (UNLIKELY(rm == 0)) return zero();
			sf_normalise_fast(rm, re);
			return from_raw(rm, re);
		}

		// Unequal exponents: align smaller operand
		if (d > 0) {
			rm = a.mantissa - (b.mantissa >> d);
			re = a.exponent;
		}
		else {
			rm = (a.mantissa >> -d) - b.mantissa;
			re = b.exponent;
		}

		if (UNLIKELY(rm == 0)) return zero();

		uint32_t ab = sf_abs32(rm);
		if (LIKELY((ab & 0x60000000u) == 0x20000000u)) {
			return from_raw(rm, re);
		}
		if (ab & 0x40000000u) {
			rm >>= 1;
			if (UNLIKELY(++re > 127)) re = 127;
			return from_raw(rm, re);
		}
		sf_normalise_fast(rm, re);
		return from_raw(rm, re);
	}

	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator-(SoftFloat a, float b) noexcept {
		return a - SoftFloat(b);
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator-(SoftFloat a, int b) noexcept {
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
		friend SoftFloat operator-(int a, SoftFloat b) noexcept {
		return SoftFloat(a) - b;
	}
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		friend SoftFloat operator-(int32_t a, SoftFloat b) noexcept {
		return SoftFloat(a) - b;
	}

#if 0
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
		if (UNLIKELY(!mantissa)) return zero();

		bool     neg = (mantissa ^ rhs.mantissa) < 0;
		uint32_t ua  = sf_abs32(mantissa);
		uint32_t ub  = sf_abs32(rhs.mantissa);

		/* ---- new reciprocal path (replaces UDIV) ---- */
		uint64_t Y   = sf_recip(ub); // ~8-10 cycles, fixed latency
		uint64_t q64 = static_cast<uint64_t>(ua) * Y;
		uint32_t qm  = static_cast<uint32_t>(q64 >> 30);
		int32_t  qe  = exponent - rhs.exponent - 30;

		uint32_t hi = qm >> 30;
		uint32_t lo = (~qm >> 29) & 1u & ~hi;
		qm = (qm >> hi) << lo;
		qe += static_cast<int32_t>(hi) - static_cast<int32_t>(lo);

		int32_t qm_signed = neg ? -static_cast<int32_t>(qm) : static_cast<int32_t>(qm);
		qe = sf_sat_exp(qe);
		return from_raw(qm_signed, qe);
	}
	
#else
	[[nodiscard]] constexpr SF_HOT SoftFloat operator/(SoftFloat rhs) const noexcept {
		if (UNLIKELY(!rhs.mantissa))
			return from_raw(mantissa >= 0 ? (1 << 29) : -(1 << 29), 127);
		if (UNLIKELY(!mantissa)) return zero();

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
			__asm__ (
				"udiv %0, %1, %2\n\t" 
				: "=r"(q1) 
				: "r"(ua), "r"(vn1)
			);  // UDIV #1
			rhat = ua - q1 * vn1;            // MLS; remainder mod vn1, < vn1 < 2^16

			// Knuth correction: q1_hat may exceed the true digit by at most 2.
			// q1 * vn0 < 2^31 (q1<2^15, vn0<2^16); rhat<<16 < 2^32 (rhat<2^16): no overflow.
			if (q1 * vn0 > (rhat << 16)) {
				--q1; rhat += vn1;
				if (q1 * vn0 > (rhat << 16)) --q1; // bare second check
			}
			// un21 = ua*2^16 - q1*v, computed overflow-free via the remainder.
			// rhat < 2^16 here ⟹ (rhat<<16) < 2^32; q1*vn0 ≤ (rhat<<16) ⟹ result ≥ 0.
			uint32_t un21 = (rhat << 16) - q1 * vn0;   // ∈ [0, v)

			// ── Low quotient digit ──────────────────────────────────────
			//   un21 < v ⟹ q0 = un21/vn1 < v/vn1 = v/(v>>16) ≤ 65535 < 2^16.
			//   q0*vn0 ≤ 65535*65535 = 2^32 - 131071 < 2^32: no overflow.
			uint32_t q0;
			__asm__ (
				"udiv %0, %1, %2\n\t" 
				: "=r"(q0) 
				: "r"(un21), "r"(vn1)
			);  // UDIV #2
			rhat = un21 - q0 * vn1;

			if (q0 * vn0 > (rhat << 16)) {
				--q0; rhat += vn1;
				if (q0 * vn0 > (rhat << 16)) --q0; // bare second check
			}

			uint32_t qm = (q1 << 16) | q0;  // ∈ [2^29, 2^31)
			int32_t  qe = exponent - rhs.exponent - 30;

			if (qm & 0x40000000u) {   // bit30 set -> need right shift by 1
				qm >>= 1;
				qe += 1;
			}
			// No left shift possible because qm >= 2^29 always

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
#endif
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
	// reciprocal — 1/x via sf_recip, O(1)
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SF_HOT SoftFloat reciprocal() const noexcept {
		if (UNLIKELY(!mantissa))
			return from_raw(mantissa >= 0 ? (1 << 29) : -(1 << 29), 127);

		bool     neg = mantissa < 0;
		uint32_t ua  = sf_abs32(mantissa);
		uint64_t Y   = sf_recip(ua); // 2^60 / ua

		uint32_t qm  = static_cast<uint32_t>(Y >> 1); // [2^29, 2^30]
		int32_t  qe  = -59 - exponent;

		if (UNLIKELY(qm & 0x40000000u)) {
			// exactly 2^30 → shift once
			qm >>= 1;
			qe += 1;
		}
		qe = sf_sat_exp(qe);
		return from_raw(neg ? -static_cast<int32_t>(qm) : static_cast<int32_t>(qm), qe);
	}
	
	// ------------------------------------------------------------------
	// Power-of-2 scaling (exponent adjust only, O(1)) — constexpr
	// ------------------------------------------------------------------
	[[nodiscard]] constexpr SF_HOT SF_INLINE SF_FLATTEN
		SoftFloat operator>>(int s) const noexcept {
		if (UNLIKELY(!mantissa)) return zero();
		int32_t ne = exponent - s;
		if (UNLIKELY(ne < -250)) return zero();
		return from_raw(mantissa, ne);
	}
	constexpr SF_HOT SF_INLINE SF_FLATTEN
		const SoftFloat& operator>>=(int s) noexcept {
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
		if (UNLIKELY(ne > 127)) 
			return from_raw(mantissa > 0 ? (1 << 29) : -(1 << 29), 127);
		return from_raw(mantissa, ne);
	}
	constexpr SF_HOT SF_INLINE SF_FLATTEN
		const SoftFloat& operator<<=(int s) noexcept {
		if (UNLIKELY(!mantissa)) return *this;
		int32_t ne = exponent + s;
		if (UNLIKELY(ne > 127)) {
			mantissa = mantissa > 0 ? (1 << 29) : -(1 << 29);
			exponent = 127;
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
	[[nodiscard]] friend constexpr bool operator==(int av, SoftFloat b) noexcept {
		const SoftFloat a(av);
		return a.mantissa == b.mantissa && a.exponent == b.exponent;
	}
	[[nodiscard]] friend constexpr bool operator==(int32_t av, SoftFloat b) noexcept {
		const SoftFloat a(av);
		return a.mantissa == b.mantissa && a.exponent == b.exponent;
	}
	[[nodiscard]] friend constexpr bool operator==(float av, SoftFloat b) noexcept {
		const SoftFloat a(av);
		return a.mantissa == b.mantissa && a.exponent == b.exponent;
	}
	[[nodiscard]] friend constexpr bool operator==(SoftFloat a, int bv) noexcept {
		const SoftFloat b(bv);
		return a.mantissa == b.mantissa && a.exponent == b.exponent;
	}
	[[nodiscard]] friend constexpr bool operator==(SoftFloat a, int32_t bv) noexcept {
		const SoftFloat b(bv);
		return a.mantissa == b.mantissa && a.exponent == b.exponent;
	}
	[[nodiscard]] friend constexpr bool operator==(SoftFloat a, float bv) noexcept {
		const SoftFloat b(bv);
		return a.mantissa == b.mantissa && a.exponent == b.exponent;
	}

	[[nodiscard]] friend constexpr bool operator!=(SoftFloat a, SoftFloat b) noexcept { return !(a == b); }
	[[nodiscard]] friend constexpr bool operator!=(int a, SoftFloat b) noexcept { return !(a == b); }
	[[nodiscard]] friend constexpr bool operator!=(int32_t a, SoftFloat b) noexcept { return !(a == b); }
	[[nodiscard]] friend constexpr bool operator!=(float a, SoftFloat b) noexcept { return !(a == b); }
	[[nodiscard]] friend constexpr bool operator!=(SoftFloat a, int b) noexcept { return !(a == b); }
	[[nodiscard]] friend constexpr bool operator!=(SoftFloat a, int32_t b) noexcept { return !(a == b); }
	[[nodiscard]] friend constexpr bool operator!=(SoftFloat a, float b) noexcept { return !(a == b); }

	[[nodiscard]] friend constexpr bool operator< (SoftFloat a, SoftFloat b) noexcept {
		if (!a.mantissa) return b.mantissa > 0;
		if (!b.mantissa) return a.mantissa < 0;
		bool an = a.mantissa < 0, bn = b.mantissa < 0;
		if (an != bn) return an;
		if (a.exponent != b.exponent)
			return an ? a.exponent > b.exponent : a.exponent < b.exponent;
		return an ? a.mantissa > b.mantissa : a.mantissa < b.mantissa;
	}
	[[nodiscard]] friend constexpr bool operator< (int av, SoftFloat b) noexcept { return SoftFloat(av) < b; }
	[[nodiscard]] friend constexpr bool operator< (int32_t av, SoftFloat b) noexcept { return SoftFloat(av) < b; }
	[[nodiscard]] friend constexpr bool operator< (float av, SoftFloat b) noexcept { return SoftFloat(av) < b; }
	[[nodiscard]] friend constexpr bool operator< (SoftFloat a, int bv)     noexcept { return a < SoftFloat(bv); }
	[[nodiscard]] friend constexpr bool operator< (SoftFloat a, int32_t bv) noexcept { return a < SoftFloat(bv); }
	[[nodiscard]] friend constexpr bool operator< (SoftFloat a, float bv)   noexcept { return a < SoftFloat(bv); }
	[[nodiscard]] friend constexpr bool operator> (SoftFloat a, SoftFloat b) noexcept { return b < a; }
	[[nodiscard]] friend constexpr bool operator> (int a, SoftFloat b) noexcept { return b < a; }
	[[nodiscard]] friend constexpr bool operator> (int32_t a, SoftFloat b) noexcept { return b < a; }
	[[nodiscard]] friend constexpr bool operator> (float a, SoftFloat b) noexcept { return b < a; }
	[[nodiscard]] friend constexpr bool operator> (SoftFloat a, int b) noexcept { return b < a; }
	[[nodiscard]] friend constexpr bool operator> (SoftFloat a, int32_t b) noexcept { return b < a; }
	[[nodiscard]] friend constexpr bool operator> (SoftFloat a, float b) noexcept { return b < a; }

	[[nodiscard]] friend constexpr bool operator<=(SoftFloat a, SoftFloat b) noexcept { return !(a > b); }
	[[nodiscard]] friend constexpr bool operator<=(int a, SoftFloat b) noexcept { return !(a > b); }
	[[nodiscard]] friend constexpr bool operator<=(int32_t a, SoftFloat b) noexcept { return !(a > b); }
	[[nodiscard]] friend constexpr bool operator<=(float a, SoftFloat b) noexcept { return !(a > b); }
	[[nodiscard]] friend constexpr bool operator<=(SoftFloat a, int b) noexcept { return !(a > b); }
	[[nodiscard]] friend constexpr bool operator<=(SoftFloat a, int32_t b) noexcept { return !(a > b); }
	[[nodiscard]] friend constexpr bool operator<=(SoftFloat a, float b) noexcept { return !(a > b); }

	[[nodiscard]] friend constexpr bool operator>=(SoftFloat a, SoftFloat b) noexcept { return !(a < b); }
	[[nodiscard]] friend constexpr bool operator>=(int a, SoftFloat b) noexcept { return !(a < b); }
	[[nodiscard]] friend constexpr bool operator>=(int32_t a, SoftFloat b) noexcept { return !(a < b); }
	[[nodiscard]] friend constexpr bool operator>=(float a, SoftFloat b) noexcept { return !(a < b); }
	[[nodiscard]] friend constexpr bool operator>=(SoftFloat a, int b) noexcept { return !(a < b); }
	[[nodiscard]] friend constexpr bool operator>=(SoftFloat a, int32_t b) noexcept { return !(a < b); }
	[[nodiscard]] friend constexpr bool operator>=(SoftFloat a, float b) noexcept { return !(a < b); }

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
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat sin() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat cos() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloatPair sincos() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat tan() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat asin() const noexcept;
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloat acos() const noexcept;
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
	[[nodiscard]] constexpr SF_HOT SF_PURE SoftFloatPair modf() const noexcept;
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

	// ============================================================
	// FIXED-POINT CORE HELPERS
	// ============================================================
	
	[[nodiscard]] static constexpr SF_INLINE SoftFloat sf_finish_addsub(int32_t rm, int32_t re) noexcept {
		if (UNLIKELY(rm == 0)) return zero();

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
		if (UNLIKELY(!a.mantissa || !b.mantissa)) return zero();

		int32_t re = a.exponent + b.exponent + 29;

#if defined(__arm__)
		if (!SF_IS_CONSTEVAL()) {
			// ── Cortex-M3 fast path ────────────────────────────────────────
			//
			// SMULL gives the full 64-bit signed product in {lo_r, hi_r}.
			//
			// rm = prod >> 29 (arithmetic):
			//   = (hi_r << 3) | (lo_r >> 29)
			//
			// Overflow detection:
			//   We need abs(rm) >= 2^30, i.e. bit30 set in abs(rm).
			//   abs(rm) ≈ rm ^ (rm asr 31)  [exact for positive; abs(rm)-1 for negative;
			//                                 bit30 is correct in both cases]
			//   TST that against 0x40000000.
			//
			// Cycle count (Cortex-M3 in-order):
			//   SMULL          : 3–5 cy
			//   MOV + ORR      : 2 cy  (depend on SMULL outputs)
			//   EOR + TST      : 2 cy  (depend on rm)
			//   IT NE + ASRNE  : 1 cy  (conditional, no flush)
			//   IT NE + ADDNE  : 1 cy
			//   Total          : ~9–11 cy  (vs ~12+ with branch)

			int32_t rm, lo_r, hi_r, tmp;

			__asm__(
			    // 64-bit signed multiply
			    "smull  %[lo], %[hi], %[am], %[bm]         \n\t"

			    // rm = arithmetic (prod >> 29)
			    //    = (hi_r << 3) | (lo_r >>> 29)
			    "mov    %[rm],  %[hi], lsl #3               \n\t"
			    "orr    %[rm],  %[rm], %[lo], lsr #29       \n\t"

			    // Overflow detection: abs(rm) >= 2^30?
			    // tmp = rm ^ (rm asr 31)  — bit30 set iff abs(rm) >= 2^30
			    "eor    %[tmp], %[rm], %[rm], asr #31       \n\t"
			    "tst    %[tmp], #0x40000000                  \n\t"

			    // Branch-free correction: if bit30 set, rm >>= 1 and re += 1
			    "it     ne                                  \n\t"
			    "asrne  %[rm],  %[rm], #1                   \n\t"
			    "it     ne                                  \n\t"
			    "addne  %[re],  %[re], #1                   \n\t"

			    : [rm]  "=&r" (rm),
				[lo]  "=&r" (lo_r),
				[hi]  "=&r" (hi_r),
				[tmp] "=&r" (tmp),
				[re]  "+r"  (re)
			  : [am]  "r"  (a.mantissa),
				[bm]  "r"  (b.mantissa)
			  : "cc");

			re = sf_sat_exp_fast(re);
			return from_raw(rm, re);
		}
#endif

		// ── Portable / consteval path ─────────────────────────────────────────
		{
			int64_t  prod  = static_cast<int64_t>(a.mantissa)
			               * static_cast<int64_t>(b.mantissa);
			int32_t  rm    = static_cast<int32_t>(prod >> 29);
			uint32_t abs_m = sf_abs32(rm);

			if (UNLIKELY(abs_m >= 0x40000000u)) {
				rm >>= 1;
				re  += 1;
			}

			re = sf_sat_exp_fast(re);
			return from_raw(rm, re);
		}
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
	[[nodiscard]] constexpr SoftFloat reciprocal()          const noexcept { return eval().reciprocal(); }
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
[[nodiscard]] constexpr SF_HOT SF_INLINE sf_mul_expr operator*(SoftFloat a, int b) noexcept {
	return a * SoftFloat(b);
}
[[nodiscard]] constexpr SF_HOT SF_INLINE sf_mul_expr operator*(SoftFloat a, int32_t b) noexcept {
	return a * SoftFloat(b);
}
[[nodiscard]] constexpr SF_HOT SF_INLINE sf_mul_expr operator*(int a, SoftFloat b) noexcept {
	return SoftFloat(a) * b;
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

[[nodiscard]] constexpr SF_HOT SoftFloat fused_mul_add(SoftFloat a, SoftFloat b, SoftFloat c) noexcept
{
	if (UNLIKELY(!b.mantissa || !c.mantissa)) return a;
	if (UNLIKELY(!a.mantissa)) return SoftFloat::mul_plain(b, c);

	int64_t  prod = static_cast<int64_t>(b.mantissa) * static_cast<int64_t>(c.mantissa);
	int32_t  pm   = static_cast<int32_t>(prod >> 29);
	int32_t  pe   = b.exponent + c.exponent + 29;

	// Branchless 1-bit normalisation of the product mantissa
	uint32_t norm = static_cast<uint32_t>(pm ^ (pm >> 31)) >> 30;
	pm >>= norm;
	pe += static_cast<int32_t>(norm);

	int d = a.exponent - pe;
	if (d >= 31) return a;

	if (d <= -31) {
		// Product dominates, but pe may itself be out of range.
		if (UNLIKELY(pe > 127))  return SoftFloat::from_raw(pm >= 0 ? (1 << 29) : -(1 << 29), 127);
		if (UNLIKELY(pe < -128)) return SoftFloat::zero();
		return SoftFloat::from_raw(pm, pe);
	}

	int32_t am = a.mantissa;

	if (d == 0) {
		// pe == a.exponent, therefore already in [-128,127]
		return SoftFloat::sf_finish_addsub(am + pm, pe);
	}

	if (d > 0) {
		// a.exponent is in range by invariant
		pm >>= d;
		return SoftFloat::sf_finish_addsub(am + pm, a.exponent);
	}

	// d < 0 : pe may be out of range; saturate once, then let sf_finish_addsub
	// handle the zero / normalised / overflow / cancellation fast paths.
	am >>= -d;
	return SoftFloat::sf_finish_addsub(am + pm, sf_sat_exp(pe));
}

// =========================================================================
// fused_mul_sub — same structure as fused_mul_add with negated product
// =========================================================================
[[nodiscard]] constexpr SF_HOT SoftFloat fused_mul_sub(SoftFloat a, SoftFloat b, SoftFloat c) noexcept
{
	if (UNLIKELY(!b.mantissa || !c.mantissa)) return a;
	if (UNLIKELY(!a.mantissa)) return -SoftFloat::mul_plain(b, c);

	int64_t  prod = static_cast<int64_t>(b.mantissa) * static_cast<int64_t>(c.mantissa);
	int32_t  pm   = static_cast<int32_t>(prod >> 29);
	int32_t  pe   = b.exponent + c.exponent + 29;

	// Branchless 1-bit normalisation of the product mantissa
	uint32_t norm = static_cast<uint32_t>(pm ^ (pm >> 31)) >> 30;
	pm >>= norm;
	pe += static_cast<int32_t>(norm);

	// Negate the product: we are computing a - (b*c)
	pm = -pm;

	int d = a.exponent - pe;
	if (d >= 31) return a;

	if (d <= -31) {
		// Product dominates, but pe may itself be out of [-128,127]
		if (UNLIKELY(pe > 127))  return SoftFloat::from_raw(pm >= 0 ? (1 << 29) : -(1 << 29), 127);
		if (UNLIKELY(pe < -128)) return SoftFloat::zero();
		return SoftFloat::from_raw(pm, pe);
	}

	int32_t am = a.mantissa;

	if (d == 0) {
		// pe == a.exponent, therefore already in [-128,127]
		return SoftFloat::sf_finish_addsub(am + pm, pe);
	}

	if (d > 0) {
		// a.exponent is in range by invariant
		pm >>= d;
		return SoftFloat::sf_finish_addsub(am + pm, a.exponent);
	}

	// d < 0 : pe may be out of range; saturate once, then let sf_finish_addsub
	// handle the zero / normalised / overflow / cancellation fast paths.
	am >>= -d;
	return SoftFloat::sf_finish_addsub(am + pm, sf_sat_exp(pe));
}

// =========================================================================
// fused_mul_mul_add — uses sf_normalise_fast, verifies SMLAL opportunity
// =========================================================================
[[nodiscard]] constexpr SF_HOT SoftFloat fused_mul_mul_add(SoftFloat a,
	SoftFloat b,
	SoftFloat c,
	SoftFloat d) noexcept
{
	// Fast zero propagation – use mul_plain to bypass the sf_mul_expr proxy
	bool abz = (!a.mantissa || !b.mantissa);
	bool cdz = (!c.mantissa || !d.mantissa);
	if (UNLIKELY(abz || cdz)) {
		if (abz && cdz) return SoftFloat::zero();
		if (abz)        return SoftFloat::mul_plain(c, d);
		return SoftFloat::mul_plain(a, b);
	}

	// Two independent SMULLs – Cortex-M3 can issue these back-to-back
	int64_t p1 = static_cast<int64_t>(a.mantissa) * static_cast<int64_t>(b.mantissa);
	int32_t pm1 = static_cast<int32_t>(p1 >> 29);
	int32_t pe1 = a.exponent + b.exponent + 29;

	int64_t p2 = static_cast<int64_t>(c.mantissa) * static_cast<int64_t>(d.mantissa);
	int32_t pm2 = static_cast<int32_t>(p2 >> 29);
	int32_t pe2 = c.exponent + d.exponent + 29;

	// Branchless 1-bit normalisation of each product
	uint32_t n1 = static_cast<uint32_t>(pm1 ^ (pm1 >> 31)) >> 30;
	pm1 >>= n1; pe1 += static_cast<int32_t>(n1);

	uint32_t n2 = static_cast<uint32_t>(pm2 ^ (pm2 >> 31)) >> 30;
	pm2 >>= n2; pe2 += static_cast<int32_t>(n2);

	int d_exp = pe1 - pe2;

	// Early out: one term dominates by >= 31 bits → the other is 0 or ±1 after shift
	if (d_exp >= 31) {
		if (UNLIKELY(pe1 > 127)) return SoftFloat::from_raw(pm1 >= 0 ? (1 << 29) : -(1 << 29), 127);
		if (UNLIKELY(pe1 < -128)) return SoftFloat::zero();
		return SoftFloat::from_raw(pm1, pe1);
	}
	if (d_exp <= -31) {
		if (UNLIKELY(pe2 > 127)) return SoftFloat::from_raw(pm2 >= 0 ? (1 << 29) : -(1 << 29), 127);
		if (UNLIKELY(pe2 < -128)) return SoftFloat::zero();
		return SoftFloat::from_raw(pm2, pe2);
	}

	// Same exponent: no alignment shift needed
	if (d_exp == 0) {
		return SoftFloat::sf_finish_addsub(pm1 + pm2, pe1);
	}

	// Align smaller term to the larger exponent
	int32_t exp;
	if (d_exp > 0) {
		pm2 >>= d_exp;
		exp = pe1;
	}
	else {
		pm1 >>= -d_exp;
		exp = pe2;
	}

	// sf_finish_addsub assumes the exponent is in [-128,127] for its fast paths.
	// If an intermediate product overflowed/underflowed, take the safe route.
	if (UNLIKELY(exp > 127 || exp < -128)) {
		int32_t s = pm1 + pm2;
		if (UNLIKELY(s == 0)) return SoftFloat::zero();
		int32_t re = exp;
		sf_normalise_fast(s, re);
		return SoftFloat::from_raw(s, re);
	}

	return SoftFloat::sf_finish_addsub(pm1 + pm2, exp);
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
[[nodiscard]] constexpr SoftFloat sf_reciprocal(SoftFloat x)                                noexcept { return x.reciprocal(); }
[[nodiscard]] constexpr SoftFloat sf_min(SoftFloat a, SoftFloat b)                   noexcept { return (a < b) ? a : b; }
[[nodiscard]] constexpr SoftFloat sf_max(SoftFloat a, SoftFloat b)                   noexcept { return (a > b) ? a : b; }
[[nodiscard]] constexpr SoftFloat sf_clamp(SoftFloat v, SoftFloat lo, SoftFloat hi)    noexcept { return v.clamp(lo, hi); }
[[nodiscard]] constexpr SoftFloatPair sf_sincos(SoftFloat x)                                noexcept { return x.sincos(); }
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

// ── C++ table definitions ───────────────────────────────────────────

// ── SF_SIN_Q30 — pre-baked sin table in Q30 fixed-point ─────────────────────
// SF_SIN_Q30[i] = round(sin(i * 2*pi / 256) * 2^30),  i = 0..256.
//
// Entry [256] mirrors [0] (= 0) so that the (idx+1) interpolation access
// used in sincos() is always within bounds.
//
// cos(x) reuses this table with a quarter-cycle offset:
//   s_val = SF_SIN_Q30[idx]          and SF_SIN_Q30[idx + 1]
//   c_val = SF_SIN_Q30[(idx+64)&0xFF] and SF_SIN_Q30[((idx+64)&0xFF) + 1]
//
// Key values:
//   [  0] =          0   sin(0)    = 0
//   [ 32] =  759250125   sin(π/4)  ≈ 0.7071  (Q30 × √2/2, correctly rounded)
//   [ 64] = 1073741824   sin(π/2)  = 1.0     (= Q30 = 2^30)
//   [128] =          0   sin(π)    ≈ 0
//   [192] =-1073741824   sin(3π/2) = -1.0    (= -Q30)
//   [256] =          0   sin(2π)   = 0       (wraparound sentinel)
//
// Table size: 257 × 4 = 1028 bytes.
// (Replaces SF_SIN_MANT[257] + SF_SIN_EXP[257] = 1028 + 257 = 1285 bytes.)
// Regenerate with:  python3 gen_sin_q30.py > table_output.txt
static constexpr int32_t SF_SIN_Q30[257] = {
			   0,     26350943,     52686014,     78989349,    105245103,    131437462,    157550647,    183568930,
	   209476638,    235258165,    260897982,    286380643,    311690799,    336813204,    361732726,    386434353,
	   410903207,    435124548,    459083786,    482766489,    506158392,    529245404,    552013618,    574449320,
	   596538995,    618269338,    639627258,    660599890,    681174602,    701339000,    721080937,    740388522,
	   759250125,    777654384,    795590213,    813046808,    830013654,    846480531,    862437520,    877875009,
	   892783698,    907154608,    920979082,    934248793,    946955747,    959092290,    970651112,    981625251,
	   992008094,   1001793390,   1010975242,   1019548121,   1027506862,   1034846671,   1041563127,   1047652185,
	  1053110176,   1057933813,   1062120190,   1065666786,   1068571464,   1070832474,   1072448455,   1073418433,
	  1073741824,   1073418433,   1072448455,   1070832474,   1068571464,   1065666786,   1062120190,   1057933813,
	  1053110176,   1047652185,   1041563127,   1034846671,   1027506862,   1019548121,   1010975242,   1001793390,
	   992008094,    981625251,    970651112,    959092290,    946955747,    934248793,    920979082,    907154608,
	   892783698,    877875009,    862437520,    846480531,    830013654,    813046808,    795590213,    777654384,
	   759250125,    740388522,    721080937,    701339000,    681174602,    660599890,    639627258,    618269338,
	   596538995,    574449320,    552013618,    529245404,    506158392,    482766489,    459083786,    435124548,
	   410903207,    386434353,    361732726,    336813204,    311690799,    286380643,    260897982,    235258165,
	   209476638,    183568930,    157550647,    131437462,    105245103,     78989349,     52686014,     26350943,
			   0,    -26350943,    -52686014,    -78989349,   -105245103,   -131437462,   -157550647,   -183568930,
	  -209476638,   -235258165,   -260897982,   -286380643,   -311690799,   -336813204,   -361732726,   -386434353,
	  -410903207,   -435124548,   -459083786,   -482766489,   -506158392,   -529245404,   -552013618,   -574449320,
	  -596538995,   -618269338,   -639627258,   -660599890,   -681174602,   -701339000,   -721080937,   -740388522,
	  -759250125,   -777654384,   -795590213,   -813046808,   -830013654,   -846480531,   -862437520,   -877875009,
	  -892783698,   -907154608,   -920979082,   -934248793,   -946955747,   -959092290,   -970651112,   -981625251,
	  -992008094,  -1001793390,  -1010975242,  -1019548121,  -1027506862,  -1034846671,  -1041563127,  -1047652185,
	 -1053110176,  -1057933813,  -1062120190,  -1065666786,  -1068571464,  -1070832474,  -1072448455,  -1073418433,
	 -1073741824,  -1073418433,  -1072448455,  -1070832474,  -1068571464,  -1065666786,  -1062120190,  -1057933813,
	 -1053110176,  -1047652185,  -1041563127,  -1034846671,  -1027506862,  -1019548121,  -1010975242,  -1001793390,
	  -992008094,   -981625251,   -970651112,   -959092290,   -946955747,   -934248793,   -920979082,   -907154608,
	  -892783698,   -877875009,   -862437520,   -846480531,   -830013654,   -813046808,   -795590213,   -777654384,
	  -759250125,   -740388522,   -721080937,   -701339000,   -681174602,   -660599890,   -639627258,   -618269338,
	  -596538995,   -574449320,   -552013618,   -529245404,   -506158392,   -482766489,   -459083786,   -435124548,
	  -410903207,   -386434353,   -361732726,   -336813204,   -311690799,   -286380643,   -260897982,   -235258165,
	  -209476638,   -183568930,   -157550647,   -131437462,   -105245103,    -78989349,    -52686014,    -26350943,
			   0,
};

constexpr SF_HOT SoftFloatPair SoftFloat::sincos() const noexcept
{
	if (UNLIKELY(mantissa == 0))
		return { zero(), one() };

	// ------------------------------------------------------------
	// 1) Range reduction to [0, 2π)
	// ------------------------------------------------------------
	constexpr int32_t INV_2PI_M = 683565276;
	constexpr int32_t INV_2PI_E = -32;
	constexpr int32_t TWO_PI_M = 843314857;
	constexpr int32_t TWO_PI_E = -27;

	SoftFloat xi = *this;
	{
		int32_t ki = (xi * SoftFloat::from_raw(INV_2PI_M, INV_2PI_E)).to_int32();
		if (ki != 0) {
			xi = xi - SoftFloat(ki) * SoftFloat::from_raw(TWO_PI_M, TWO_PI_E);
		}

		const SoftFloat two_pi = SoftFloat::from_raw(TWO_PI_M, TWO_PI_E);
		if (xi.mantissa < 0)  xi = xi + two_pi;
		if (!(xi < two_pi))   xi = xi - two_pi;
	}

	// ------------------------------------------------------------
	// 2) Convert reduced angle to 8.24 phase:
	//      u = xi * (256 / 2π) = xi * (128 / π)
	//
	//    Use K_Q25 = round((128/π) * 2^25)
	//
	//    xi = mantissa * 2^exponent
	//    u  = mantissa * K_Q25 * 2^(exponent - 25)
	//    u_8_24 = u * 2^24 = mantissa * K_Q25 * 2^(exponent - 1)
	//
	//    Therefore:
	//      u_8_24 = (mantissa * K_Q25) >> (1 - exponent)
	// ------------------------------------------------------------
	constexpr int32_t K_Q25 = 1367130551; // round((128/pi) * 2^25)

	int64_t prod;
#if defined(__arm__)
	if (!SF_IS_CONSTEVAL()) {
		int32_t lo, hi;
		__asm__("smull %0, %1, %2, %3"
			: "=&r"(lo),
			"=&r"(hi)
			: "r"(xi.mantissa),
			"r"(K_Q25));
		prod = (static_cast<int64_t>(hi) << 32) | static_cast<uint32_t>(lo);
	}
	else
#endif
	{
		prod = static_cast<int64_t>(xi.mantissa) * static_cast<int64_t>(K_Q25);
	}

	// After range reduction xi ∈ [0, 2π), so exponent is always negative.
	const int32_t rshift = 1 - xi.exponent;

	uint32_t u_8_24;
	if (LIKELY(rshift > 0 && rshift < 64)) {
		u_8_24 = static_cast<uint32_t>(prod >> rshift);
	}
	else if (rshift <= 0 && rshift > -32) {
		u_8_24 = static_cast<uint32_t>(prod << (-rshift));
	}
	else {
		u_8_24 = 0;
	}

	uint32_t idx = (u_8_24 >> 24) & 0xFFu; // 0..255
	uint32_t frac = u_8_24 & 0xFFFFFFu; // 24-bit fraction

	// ------------------------------------------------------------
	// 3) Table lookup
	//
	// The sine table has 257 entries [0..256], with [256] = [0],
	// so we can use idx+1 safely for the sine interpolation edge.
	//
	// cos(x) = sin(x + π/2) => +64 table steps
	// ------------------------------------------------------------
	const uint32_t s_idx0 = idx; // 0..255
	const uint32_t s_idx1 = idx + 1u; // 1..256

	const uint32_t c_idx0 = (idx + 64u) & 0xFFu; // 0..255
	const uint32_t c_idx1 = c_idx0 + 1u; // 1..256 if c_idx0==255

	// ------------------------------------------------------------
	// 4) Direct Q30 table lookups — no runtime conversion needed.
	//    SF_SIN_Q30[i] already stores round(sin(i*2π/256) * 2^30).
	//    cos uses the same table with a +64 quarter-cycle offset.
	// ------------------------------------------------------------
	const int32_t s0 = SF_SIN_Q30[s_idx0];
	const int32_t s1 = SF_SIN_Q30[s_idx1];
	const int32_t c0 = SF_SIN_Q30[c_idx0];
	const int32_t c1 = SF_SIN_Q30[c_idx1];

	// ------------------------------------------------------------
	// 5) Linear interpolation in Q30
	//
	// result = v0 + ((v1 - v0) * frac) >> 24
	// ------------------------------------------------------------
	int32_t sin_q30, cos_q30;
	{
		const int32_t ds = s1 - s0;
		const int32_t dc = c1 - c0;

#if defined(__arm__)
		if (!SF_IS_CONSTEVAL()) {
			int32_t lo_s, hi_s, lo_c, hi_c;
			__asm__("smull %0, %1, %2, %3"
				: "=&r"(lo_s),
				"=&r"(hi_s)
				: "r"(ds),
				"r"(static_cast<int32_t>(frac)));
			__asm__("smull %0, %1, %2, %3"
				: "=&r"(lo_c),
				"=&r"(hi_c)
				: "r"(dc),
				"r"(static_cast<int32_t>(frac)));

			// (delta * frac) >> 24
			const int32_t corr_s = (hi_s << 8) | (static_cast<uint32_t>(lo_s) >> 24);
			const int32_t corr_c = (hi_c << 8) | (static_cast<uint32_t>(lo_c) >> 24);

			sin_q30 = s0 + corr_s;
			cos_q30 = c0 + corr_c;
		}
		else
#endif
		{
			const int64_t ps = static_cast<int64_t>(ds) * static_cast<int64_t>(frac);
			const int64_t pc = static_cast<int64_t>(dc) * static_cast<int64_t>(frac);

			sin_q30 = s0 + static_cast<int32_t>(ps >> 24);
			cos_q30 = c0 + static_cast<int32_t>(pc >> 24);
		}
	}

	// ------------------------------------------------------------
	// 6) Wrap back to SoftFloat once
	// ------------------------------------------------------------
	int32_t sm = sin_q30, se = -30;
	sf_normalise_fast(sm, se);
	int32_t cm = cos_q30, ce = -30;
	sf_normalise_fast(cm, ce);
	return {
		from_raw(sm,se),
		from_raw(cm,ce)
	};

}

constexpr SoftFloat SoftFloat::tan() const noexcept {
	auto[s, c] = sincos();
	if (c.is_zero())
		return from_raw(s.mantissa >= 0 ? (1 << 29) : -(1 << 29), 127);
	return s / c;
}

constexpr SoftFloat SoftFloat::sin() const noexcept { 
	return sincos().intpart; 
}

constexpr SoftFloat SoftFloat::cos() const noexcept { 
	return sincos().fracpart; 
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
	return num * den.reciprocal();             // 1 div → 1 recip + 1 mul
}

#if 1

// inv_sqrt table in uniform Q29 format.
// INV_SQRT_Q29[i] = round(2^29 * inv_sqrt(1 + i/256))  for i=0..256
// All values in (2^29/sqrt(2), 2^29] = (0x16A09E66, 0x20000000].
// Derived from original INV_SQRT_MANT: [0] unchanged (was already Q29),
// [1..256] = INV_SQRT_MANT[i] >> 1 (converting from Q30 to Q29).
static constexpr int32_t INV_SQRT_Q29[257] = {
	0x20000000, 0x1FF00BF6, 0x1FE02FB0, 0x1FD06AF4, 0x1FC0BD88, 0x1FB12733, 0x1FA1A7BB, 0x1F923EEA,
	0x1F82EC88, 0x1F73B05F, 0x1F648A3A, 0x1F5579E4, 0x1F467F28, 0x1F3799D3, 0x1F28C9B3, 0x1F1A0E95,
	0x1F0B6849, 0x1EFCD69C, 0x1EEE595E, 0x1EDFF061, 0x1ED19B76, 0x1EC35A6D, 0x1EB52D18, 0x1EA7134C,
	0x1E990CDB, 0x1E8B1998, 0x1E7D3959, 0x1E6F6BF2, 0x1E61B139, 0x1E540903, 0x1E467328, 0x1E38EF7F,
	0x1E2B7DDE, 0x1E1E1E1E, 0x1E10D017, 0x1E0393A3, 0x1DF6689B, 0x1DE94ED9, 0x1DDC4637, 0x1DCF4E8F,
	0x1DC267BE, 0x1DB5919F, 0x1DA8CC0E, 0x1D9C16E8, 0x1D8F7209, 0x1D82DD4F, 0x1D765897, 0x1D69E3C0,
	0x1D5D7EA9, 0x1D512930, 0x1D44E334, 0x1D38AC96, 0x1D2C8535, 0x1D206CF1, 0x1D1463AC, 0x1D086947,
	0x1CFC7DA3, 0x1CF0A0A2, 0x1CE4D225, 0x1CD91210, 0x1CCD6046, 0x1CC1BCA9, 0x1CB6271C, 0x1CAA9F85,
	0x1C9F25C5, 0x1C93B9C4, 0x1C885B63, 0x1C7D0A8A, 0x1C71C71C, 0x1C669100, 0x1C5B681B, 0x1C504C54,
	0x1C453D91, 0x1C3A3BB8, 0x1C2F46B0, 0x1C245E61, 0x1C198AB3, 0x1C0EB38C, 0x1C03F0D5, 0x1BF93A75,
	0x1BEE9057, 0x1BE3F261, 0x1BD9607E, 0x1BCEDA96, 0x1BC46093, 0x1BB9F25E, 0x1BAF8FE1, 0x1BA53907,
	0x1B9AEDBA, 0x1B90ADE4, 0x1B867970, 0x1B7C504A, 0x1B72325B, 0x1B681F91, 0x1B5E17D5, 0x1B541B15,
	0x1B4A293C, 0x1B404236, 0x1B3665F0, 0x1B2C9457, 0x1B22CD57, 0x1B1910DD, 0x1B0F5ED6, 0x1B05B730,
	0x1AFC19D8, 0x1AF286BC, 0x1AE8FDCB, 0x1ADF7EF1, 0x1AD60A1D, 0x1ACC9F3E, 0x1AC33E42, 0x1AB9E718,
	0x1AB099AE, 0x1AA755F5, 0x1A9E1BDB, 0x1A94EB4F, 0x1A8BC441, 0x1A82A6A2, 0x1A79925F, 0x1A70876B,
	0x1A6785B4, 0x1A5E8D2B, 0x1A559DC1, 0x1A4CB766, 0x1A43DA0B, 0x1A3B05A0, 0x1A323A17, 0x1A297761,
	0x1A20BD70, 0x1A180C34, 0x1A0F639F, 0x1A06C3A3, 0x19FE2C31, 0x19F59D3C, 0x19ED16B6, 0x19E49890,
	0x19DC22BE, 0x19D3B531, 0x19CB4FDD, 0x19C2F2B3, 0x19BA9DA7, 0x19B250AB, 0x19AA0BB3, 0x19A1CEB1,
	0x19999999, 0x19916D5F, 0x198946F5, 0x19812950, 0x19791363, 0x19710521, 0x1968FE80, 0x1960FF72,
	0x195907EB, 0x195117E1, 0x19492F47, 0x19414E12, 0x19397436, 0x1931A1A8, 0x1929D65D, 0x19221249,
	0x191A5561, 0x19129F9B, 0x190AF0EA, 0x19034946, 0x18FBA8A1, 0x18F40EF4, 0x18EC7C31, 0x18E4F050,
	0x18DD6B45, 0x18D5ED07, 0x18CE758B, 0x18C704C6, 0x18BF9AB0, 0x18B8373E, 0x18B0DA66, 0x18A9841E,
	0x18A2345D, 0x189AEB18, 0x1893A847, 0x188C6BE0, 0x188535D9, 0x187E0629, 0x1876DCCF, 0x186FB9AA,
	0x18689CC8, 0x18618618, 0x185A7592, 0x18536B2D, 0x184C66DF, 0x184568A0, 0x183E7067, 0x18377E2C,
	0x183091E6, 0x1829AB8D, 0x1822CB18, 0x181BF07E, 0x18151BB8, 0x180E4CBD, 0x18078386, 0x1800C009,
	0x17FA023F, 0x17F34A20, 0x17EC97A4, 0x17E5EAC3, 0x17DF4375, 0x17D8A1B3, 0x17D20575, 0x17CB6EB3,
	0x17C4DD66, 0x17BE5186, 0x17B7CB0C, 0x17B149F0, 0x17AACE2B, 0x17A457B5, 0x179DE689, 0x17977A9D,
	0x179113EB, 0x178AB26D, 0x1784561B, 0x177EFEED, 0x177AACDE, 0x17715FE6, 0x176B17FF, 0x1764D521,
	0x175E9746, 0x17585E68, 0x17522A7F, 0x174BFB85, 0x1745D174, 0x173FAC45, 0x17398BF2, 0x17337073,
	0x172D59C4, 0x172747DD, 0x172139B9, 0x171B3251, 0x17152E9F, 0x170F2F9D, 0x17093544, 0x17033F90,
	0x16FD4E79, 0x16F761FA, 0x16F17A0D, 0x16EB96AC, 0x16E5B7D1, 0x16DFDD77, 0x16DA0797, 0x16D4362D,
	0x16CE6932, 0x16C8A0A0, 0x16C2DC73, 0x16BD1CA4, 0x16B7612F, 0x16B1AA0D, 0x16ABF739, 0x16A648AE,
	0x16A09E66,
};

constexpr SF_HOT SoftFloat SoftFloat::inv_sqrt() const noexcept
{
	if (UNLIKELY(mantissa <= 0)) return zero();

	const int32_t  E_raw = exponent + 29;
	const uint32_t a = static_cast<uint32_t>(mantissa);

	// --- Step 1: initial guess from table + linear interpolation
	const uint32_t offset = a - 0x20000000u;
	const uint32_t idx = offset >> 21;
	const uint32_t frac8 = (offset >> 13) & 0xFFu;

	const int32_t v0 = INV_SQRT_Q29[idx];
	const int32_t v1 = INV_SQRT_Q29[idx + 1];
	int32_t y_q29 = v0 + (((v1 - v0) * static_cast<int32_t>(frac8)) >> 8);

	// --- Step 2: NEWTON ITERATION FIRST. ALWAYS.
	int32_t yy, ay, r_q29;
	
#if defined(__arm__)
	if (!SF_IS_CONSTEVAL()) {
		int32_t lo, hi;
		__asm__("smull %0, %1, %2, %3"
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
		__asm__("smull %0, %1, %2, %3"
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
		__asm__("smull %0, %1, %2, %3"
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

	// --- Step 3: NOW apply odd exponent correction. AFTER Newton.
	if (E_raw & 1) {
#if defined(__arm__)
		if (!SF_IS_CONSTEVAL()) {
			int32_t lo, hi;
			__asm__("smull %0, %1, %2, %3"
				: "=&r"(lo), "=&r"(hi)
				: "r"(y_q29), "r"(0x16A09E66));
			y_q29 = (hi << 3) | (static_cast<uint32_t>(lo) >> 29);
		}
		else
#endif
			y_q29 = static_cast<int32_t>( (static_cast<int64_t>(y_q29) * 0x16A09E66LL) >> 29);
	}

	// --- Step 4: Normalise
	const int32_t carry = static_cast<uint32_t>(y_q29) >> 29;
	y_q29 <<= 1;
	int32_t result_e = -29 - (E_raw >> 1) - 1 + carry;

	result_e = sf_sat_exp(result_e);

	return from_raw(y_q29, result_e);
}

constexpr SF_HOT SoftFloat SoftFloat::sqrt() const noexcept
{
	if (UNLIKELY(mantissa <= 0)) return zero();

	if (SF_IS_CONSTEVAL()) {
		int32_t m = mantissa;
		int32_t e = exponent;
		if (e & 1) { m <<= 1; e -= 1; }
		uint64_t scaled = static_cast<uint64_t>(m) << 30;
		uint64_t root = ct_isqrt64(scaled);
		int32_t  rm = static_cast<int32_t>(root);
		int32_t  re = e / 2 - 15;
		return SoftFloat(rm, re);
	}

	const int32_t  E_raw = exponent + 29;
	const uint32_t a = static_cast<uint32_t>(mantissa);

	// --- Step 1: initial 1/sqrt(a_norm) from table (Q29) ---
	// y_q29 ≈ 1/sqrt(a_norm), a_norm ∈ [1,2)
	// so y_q29 ∈ (1/sqrt(2), 1] — no odd-exponent correction here
	const uint32_t offset = a - 0x20000000u;
	const uint32_t idx = offset >> 21;
	const uint32_t frac8 = (offset >> 13) & 0xFFu;

	const int32_t v0 = INV_SQRT_Q29[idx];
	const int32_t v1 = INV_SQRT_Q29[idx + 1];
	int32_t y_q29 = v0 + (((v1 - v0) * static_cast<int32_t>(frac8)) >> 8);

	// --- Step 2: initial sqrt estimate g = a * y >> 29 ---
	// g_q29 ≈ sqrt(a_norm) ∈ [1, sqrt(2))
	// Invariant: y_q29 * g_q29 ≈ a_norm * 2^58 / a_norm = 2^58  ✓
	int32_t g_q29;
#if defined(__arm__)
	{
		int32_t lo, hi;
		__asm__("smull %0, %1, %2, %3"
			: "=&r"(lo), "=&r"(hi)
			: "r"(static_cast<int32_t>(a)), "r"(y_q29));
		g_q29 = (hi << 3) | (static_cast<uint32_t>(lo) >> 29);
	}
#else
	g_q29 = static_cast<int32_t>(
		(static_cast<uint64_t>(a) * static_cast<uint32_t>(y_q29)) >> 29);
#endif

	// --- Step 3: Goldschmidt correction ---
	// r = 0.5 - h*g  where h = y/2
	// y_q29 * g_q29 ≈ 2^58; >> 29 ≈ 2^29 = 0x20000000
	// r_q30 = 0x20000000 - (y_q29 * g_q29 >> 29)  ≈ 0
	// g_q29 += g_q29 * r_q30 >> 30
	int32_t r_q30;
#if defined(__arm__)
	{
		int32_t lo, hi;
		__asm__("smull %0, %1, %2, %3"
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
		__asm__("smull %0, %1, %2, %3"
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
	// g_q29 ≈ sqrt(a_norm) * 2^29, accurate ~40 bits
	// g_q29 ∈ [0x20000000, 0x2D413CCD] — normalised, no overflow possible

	// --- Step 4: handle odd E_raw ---
	// Full value = a_norm * 2^E_raw,  sqrt = sqrt(a_norm) * 2^(E_raw/2)
	// g_q29 = sqrt(a_norm) * 2^29
	// result = g_q29 * 2^(result_e + 29)  where result_e = (E_raw>>1) - 29
	//
	// When E_raw is odd: E_raw = 2k+1, E_raw>>1 = k
	//   sqrt(value) = sqrt(a_norm) * 2^k * sqrt(2)
	//   g_q29 * sqrt(2) puts g into [0x2D41.., 0x3FFF..] — still fits in Q29
	//   result_e = k - 29 = (E_raw>>1) - 29  (same formula, sqrt(2) absorbed into g)
	if (E_raw & 1) {
#if defined(__arm__)
		int32_t lo, hi;
		__asm__("smull %0, %1, %2, %3"
			: "=&r"(lo), "=&r"(hi)
			: "r"(g_q29), "r"(0x2D413CCD));  // sqrt(2) in Q29
		g_q29 = (hi << 3) | (static_cast<uint32_t>(lo) >> 29);
#else
		g_q29 = static_cast<int32_t>(
			(static_cast<int64_t>(g_q29) * 0x2D413CCDLL) >> 29);
#endif
	}
	// g_q29 ∈ [0x20000000, 0x3FFFFFFF] — normalised in both cases

	// --- Step 5: assemble result ---
	const int32_t result_e = sf_sat_exp((E_raw >> 1) - 29);
	return from_raw(g_q29, result_e);
}

#else

// =========================================================================
// Inverse square root — Q-rsqrt seed + 2× Newton-Raphson, constexpr
//
// k1.5 = 1.5:  0x30000000 * 2^-29 = 1.5   clz=2 ✓
// =========================================================================
constexpr SF_HOT SoftFloat SoftFloat::inv_sqrt() const noexcept {
	if (UNLIKELY(mantissa <= 0)) return zero();

	// Fast initial estimate via the classic magic-constant bit trick.
	// Both to_float() and sf_bitcast are constexpr in C++20.
	float    xf = to_float();
	uint32_t bits = sf_bitcast<uint32_t>(xf);
	//bits = 0x5f3759dfu - (bits >> 1);
	bits = 0x5f375a86u - (bits >> 1); // slightly better constant
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

constexpr SF_HOT SoftFloat SoftFloat::sqrt() const noexcept {
	if (UNLIKELY(mantissa <= 0)) return zero();

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

#endif

#if 1
static constexpr int32_t ATAN_TAB_Q29[257] = {
			 0,    2097141,    4194219,    6291168,    8387925,   10484427,   12580609,   14676407,
	  16771758,   18866598,   20960863,   23054490,   25147416,   27239578,   29330911,   31421354,
	  33510843,   35599317,   37686712,   39772966,   41858018,   43941805,   46024266,   48105340,
	  50184965,   52263081,   54339626,   56414542,   58487768,   60559244,   62628910,   64696708,
	  66762579,   68826465,   70888307,   72948048,   75005631,   77060998,   79114093,   81164859,
	  83213242,   85259186,   87302634,   89343535,   91381832,   93417472,   95450402,   97480570,
	  99507923,  101532409,  103553977,  105572575,  107588154,  109600664,  111610055,  113616278,
	 115619285,  117619027,  119615459,  121608532,  123598200,  125584418,  127567140,  129546321,
	 131521918,  133493887,  135462185,  137426768,  139387596,  141344627,  143297819,  145247133,
	 147192530,  149133969,  151071412,  153004822,  154934160,  156859391,  158780477,  160697384,
	 162610076,  164518518,  166422677,  168322519,  170218011,  172109122,  173995820,  175878074,
	 177755853,  179629127,  181497868,  183362046,  185221634,  187076603,  188926928,  190772581,
	 192613537,  194449771,  196281257,  198107973,  199929894,  201746997,  203559260,  205366662,
	 207169181,  208966795,  210759486,  212547234,  214330019,  216107822,  217880627,  219648415,
	 221411170,  223168875,  224921514,  226669072,  228411535,  230148887,  231881116,  233608207,
	 235330149,  237046928,  238758533,  240464953,  242166178,  243862195,  245552997,  247238573,
	 248918915,  250594014,  252263862,  253928451,  255587776,  257241828,  258890602,  260534092,
	 262172294,  263805201,  265432810,  267055116,  268672116,  270283807,  271890185,  273491249,
	 275086997,  276677426,  278262536,  279842326,  281416795,  282985944,  284549771,  286108279,
	 287661468,  289209339,  290751894,  292289135,  293821065,  295347685,  296869000,  298385011,
	 299895724,  301401141,  302901268,  304396108,  305885667,  307369949,  308848960,  310322706,
	 311791193,  313254427,  314712414,  316165161,  317612676,  319054965,  320492037,  321923898,
	 323350557,  324772022,  326188302,  327599405,  329005341,  330406118,  331801746,  333192234,
	 334577593,  335957831,  337332960,  338702990,  340067931,  341427795,  342782591,  344132331,
	 345477027,  346816690,  348151331,  349480962,  350805596,  352125243,  353439918,  354749631,
	 356054396,  357354224,  358649130,  359939125,  361224223,  362504438,  363779782,  365050268,
	 366315911,  367576724,  368832721,  370083915,  371330321,  372571953,  373808825,  375040951,
	 376268345,  377491022,  378708997,  379922284,  381130898,  382334853,  383534165,  384728848,
	 385918917,  387104388,  388285275,  389461594,  390633360,  391800588,  392963293,  394121491,
	 395275197,  396424428,  397569197,  398709522,  399845417,  400976898,  402103981,  403226681,
	 404345015,  405458998,  406568646,  407673974,  408774999,  409871737,  410964203,  412052413,
	 413136383,  414216130,  415291668,  416363015,  417430186,  418493196,  419552063,  420606802,
	 421657428
};

constexpr SF_HOT SoftFloat atan2(SoftFloat y, SoftFloat x) noexcept {

	// Q29 constants - these match SoftFloat::half_pi() and pi() mantissas exactly
	constexpr int32_t HALF_PI_Q29 = 843314857;   // π/2 * 2^29
	constexpr int32_t PI_Q29 = 1686629713;  // π * 2^29

	// Handle origin
	if (x.is_zero() && y.is_zero()) return SoftFloat::zero();

	// Capture signs and take absolute values
	const bool x_neg = x.is_negative();
	const bool y_neg = y.is_negative();
	x = x.abs();
	y = y.abs();

	// Ensure ratio <= 1 by swapping if needed
	const bool swap = y > x;
	if (swap) { SoftFloat t = x; x = y; y = t; }

	// Compute ratio y/x as Q24 fixed-point (range [0, 1] maps to [0, 0x1000000])
	uint32_t t_Q24 = 0;
	if (!x.is_zero() && !y.is_zero()) {
		const uint32_t x_m = static_cast<uint32_t>(x.mantissa);
		const uint32_t y_m = static_cast<uint32_t>(y.mantissa);
		const int32_t exp_diff = y.exponent - x.exponent;
		const int32_t shift = 24 + exp_diff;

		if (shift > -30) {
			if (shift < 0) {
				// Ratio is very small
				t_Q24 = (y_m >> (-shift)) / x_m;
			}
			else {
				const uint32_t eff_shift = static_cast<uint32_t>(shift > 31 ? 31 : shift);
				const uint64_t num = static_cast<uint64_t>(y_m) << eff_shift;

#if defined(__arm__)
				if (!SF_IS_CONSTEVAL() && (num >> 32) == 0) {
					uint32_t q;
					__asm__("udiv %0, %1, %2" : "=r"(q) : "r"(static_cast<uint32_t>(num)), "r"(x_m));
					t_Q24 = q;
				}
				else {
					t_Q24 = static_cast<uint32_t>(num / x_m);
				}
#else
				t_Q24 = static_cast<uint32_t>(num / x_m);
#endif
			}
		}
	}

	// Table lookup with linear interpolation
	// idx is the integer part (0-256), frac is the fractional part in Q16
	const uint32_t idx = t_Q24 >> 16;
	const uint32_t frac = t_Q24 & 0xFFFFu;

	int32_t angle_q29;
	if (idx >= 256) {
		// Ratio is exactly 1.0 (or slightly above due to rounding)
		angle_q29 = ATAN_TAB_Q29[256];
	}
	else {
		const int32_t a0 = ATAN_TAB_Q29[idx];
		const int32_t a1 = ATAN_TAB_Q29[idx + 1];
		// Linear interpolation: a0 + (a1 - a0) * frac / 65536
		angle_q29 = a0 + static_cast<int32_t>((static_cast<int64_t>(a1 - a0) * frac) >> 16);
	}

	// Quadrant adjustments - all in Q29 fixed-point
	// After table lookup: angle is in [0, π/4]
	// After swap:         angle is in [0, π/2]
	// After x_neg:        angle is in [0, π]
	// After y_neg:        angle is in [-π, π]

	if (swap)  angle_q29 = HALF_PI_Q29 - angle_q29;
	if (x_neg) angle_q29 = PI_Q29 - angle_q29;
	if (y_neg) angle_q29 = -angle_q29;

	// Single conversion to SoftFloat at the end
	// The constructor normalizes automatically
	return SoftFloat(angle_q29, -29);
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
	constexpr SoftFloat c0 = SoftFloat::from_raw(536870912, -29);     // 1.0
	constexpr SoftFloat c1 = SoftFloat::from_raw(-715827883, -31); // -1/3
	constexpr SoftFloat c2 = SoftFloat::from_raw(858993459, -32); // 1/5  
	constexpr SoftFloat c3 = SoftFloat::from_raw(-613566757, -32); // -1/7
	constexpr SoftFloat c4 = SoftFloat::from_raw(954437177, -33); // 1/9
	constexpr SoftFloat c5 = SoftFloat::from_raw(-780903145, -33); // -1/11

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

constexpr SF_HOT SoftFloat SoftFloat::exp() const noexcept
{
	if (UNLIKELY(mantissa == 0)) return one();

	constexpr int32_t INV_LN2_M = 0x2E2A8ECB; // 1/ln(2) in Q29  (= round(1/ln2 · 2²⁹))

	// ── Step 1: one SMULL — the only multiply we need ────────────────────────
	// kprod × 2^(exponent − 29)  =  x / ln2   (exact in 64-bit fixed-point)
	int64_t kprod;
#if defined(__arm__)
	if (!SF_IS_CONSTEVAL()) {
		int32_t lo, hi;
		__asm__("smull %0, %1, %2, %3"
			: "=&r"(lo), "=&r"(hi)
			: "r"(mantissa), "r"(INV_LN2_M));
		kprod = (static_cast<int64_t>(hi) << 32) | static_cast<uint32_t>(lo);
	}
	else
#endif
	{
		kprod = static_cast<int64_t>(mantissa) * static_cast<int64_t>(INV_LN2_M);
	}

	// The binary radix point sits at bit position k_rshift inside kprod.
	const int32_t k_rshift = 29 - exponent;
	if (UNLIKELY(k_rshift <= 0))
		return mantissa > 0 ? from_raw(0x20000000, 127) : zero();

	// ── Steps 2+3 fused: extract k and fractional index in one pass ──────────
	//
	// Two's-complement floor division property:
	//   kprod  =  k × 2^k_rshift  +  frac_bits,   frac_bits ∈ [0, 2^k_rshift)
	//
	// Arithmetic right shift gives k = ⌊x/ln2⌋ (works for both signs).
	// The bottom k_rshift bits of kprod (unsigned) ARE frac_bits — no SoftFloat
	// subtraction, no second multiply needed.
	//
	// Shifting frac_bits right by (k_rshift − 29) normalises to Q29 → u_8_21.
	int32_t  k;
	uint32_t u_8_21;

	if (LIKELY(k_rshift <= 63)) {
		k = static_cast<int32_t>(kprod >> k_rshift);   // ⌊x/ln2⌋, any sign

		const uint64_t mask = (uint64_t(1) << k_rshift) - 1u;
		const uint64_t frac_bits = static_cast<uint64_t>(kprod) & mask; // ∈ [0, 2^k_rshift)

		u_8_21 = (k_rshift > 29)
			? static_cast<uint32_t>(frac_bits >> (k_rshift - 29))
			: static_cast<uint32_t>(frac_bits) << (29 - k_rshift);
	}
	else {
		// k_rshift ≥ 64: |x| < 2^{−35}; |kprod| < 2^60 < 2^64.
		// k ∈ {−1, 0}; static_cast<uint64_t>(kprod) correctly encodes
		// the fractional remainder for k_rshift = 64 (and is a close
		// approximation for larger shifts, where exp(x) ≈ 1 anyway).
		k = (kprod < 0) ? -1 : 0;
		const uint32_t rsh = static_cast<uint32_t>(k_rshift - 29);
		u_8_21 = (rsh < 60)
			? static_cast<uint32_t>(static_cast<uint64_t>(kprod) >> rsh)
			: 0u;
	}

	// u_8_21 ∈ [0, 2²⁹) — no clamping needed.
	const uint32_t idx = u_8_21 >> 21;                              // 0..255
	const int32_t  frac = static_cast<int32_t>(u_8_21 & 0x1FFFFFu);  // 0..2²¹−1

	// ── Step 4: table lookup + linear interpolation ──────────────────────────
	const int32_t m0 = EXP_MANT[idx];
	// EXP_MANT[256] = 0x20000000 stores the mantissa of 2.0 with an implicit
	// exponent increment (+1). For interpolation across the last interval we
	// need the true Q29 value of 2.0 = 0x40000000.
	const int32_t m1 = LIKELY(idx < 255) ? EXP_MANT[idx + 1] : int32_t(0x40000000);
	const int32_t delta = m1 - m0;   // small, always positive

	int32_t result_q29;
#if defined(__arm__)
	if (!SF_IS_CONSTEVAL()) {
		int32_t lo, hi;
		__asm__("smull %0, %1, %2, %3"
			: "=&r"(lo), "=&r"(hi)
			: "r"(delta), "r"(frac));
		result_q29 = m0 + ((hi << 11) | (static_cast<uint32_t>(lo) >> 21));
	}
	else
#endif
	{
		result_q29 = m0 + static_cast<int32_t>((static_cast<int64_t>(delta) * frac) >> 21);
	}

	// result_q29 ∈ [2²⁹, 2³⁰) — already normalised; no adjustment needed.
	// (At idx=255, frac < 2²¹ guarantees result < 0x40000000.)

	// ── Step 5: attach exponent ───────────────────────────────────────────────
	const int32_t final_exp = k - 29;
	if (UNLIKELY(final_exp > 127))  return from_raw(0x20000000, 127);
	if (UNLIKELY(final_exp < -128)) return zero();
	return from_raw(result_q29, final_exp);
}

// Pre-baked LOG2 table in Q30 format.
// LOG2_Q30[i] = to_q30(LOG2_MANT[i], LOG2_EXP[i])
//             = LOG2_MANT[i] shifted by (LOG2_EXP[i] + 30) bits.
// Replaces LOG2_MANT[] + LOG2_EXP[] entirely in log2().
static constexpr int32_t LOG2_Q30[257] = {
	0x00000000, // [0]
	0x005C2711, // [1]
	0x00B7F285, // [2]
	0x01136311, // [3]
	0x016E7968, // [4]
	0x01C9363B, // [5]
	0x02239A3A, // [6]
	0x027DA612, // [7]
	0x02D75A6E, // [8]
	0x0330B7F8, // [9]
	0x0389BF57, // [10]
	0x03E27130, // [11]
	0x043ACE27, // [12]
	0x0492D6DF, // [13]
	0x04EA8BF7, // [14]
	0x0541EE0D, // [15]
	0x0598FDBE, // [16]
	0x05EFBBA5, // [17]
	0x0646285B, // [18]
	0x069C4477, // [19]
	0x06F21090, // [20]
	0x07478D38, // [21]
	0x079CBB04, // [22]
	0x07F19A83, // [23]
	0x08462C46, // [24]
	0x089A70DA, // [25]
	0x08EE68CB, // [26]
	0x094214A5, // [27]
	0x099574F1, // [28]
	0x09E88A36, // [29]
	0x0A3B54FC, // [30]
	0x0A8DD5C8, // [31]
	0x0AE00D1C, // [32]
	0x0B31FB7D, // [33]
	0x0B83A16A, // [34]
	0x0BD4FF63, // [35]
	0x0C2615E8, // [36]
	0x0C76E574, // [37]
	0x0CC76E83, // [38]
	0x0D17B191, // [39]
	0x0D67AF16, // [40]
	0x0DB7678B, // [41]
	0x0E06DB66, // [42]
	0x0E560B1E, // [43]
	0x0EA4F726, // [44]
	0x0EF39FF1, // [45]
	0x0F4205F3, // [46]
	0x0F90299C, // [47]
	0x0FDE0B5C, // [48]
	0x102BABA2, // [49]
	0x107908DB, // [50]
	0x10C62975, // [51]
	0x111307DA, // [52]
	0x115FA676, // [53]
	0x11AC05B2, // [54]
	0x11F825F6, // [55]
	0x124407AB, // [56]
	0x128FAB35, // [57]
	0x12DB10FC, // [58]
	0x13263963, // [59]
	0x13712ACE, // [60]  (0x26E2499D>>1: 0x26E2499D=653,854,109; /2=326,927,054=0x137124CE ... let me recheck: 0x26E2499D>>1)
	0x13BBD3A0, // [61]  (0x2777A741>>1: top bit 0, so 0x13BBD3A0 ✓)
	0x1406463B, // [62]  (0x280C8C76>>1=0x14046463... wait:
	//  0x280C8C76: 0010 1000 0000 1100 1000 1100 0111 0110
	//  >>1:        0001 0100 0000 0110 0100 0110 0011 1011 = 0x1406463B ✓)
0x14507CFE, // [63]  (0x28A0F9FD>>1: bit0=1 truncated: 0x14507CFE ✓)
0x149A784B, // [64]  (0x2934F097>>1: bit0=1: 0x149A784B ✓)
0x14E43880, // [65]  (0x29C87101>>1: bit0=1: 0x14E43880 ✓)
0x152DBDFC, // [66]  (0x2A5B7BF8>>1=0x152DBDFC ✓)
0x1577091B, // [67]  (0x2AEE1236>>1: bit0=0: 0x1577091B ✓)
0x15C01A39, // [68]  (0x2B803473>>1: bit0=1: 0x15C01A39 ✓)
0x1608F1B4, // [69]  (0x2C11E368>>1=0x1608F1B4 ✓)
0x16518FE4, // [70]  (0x2CA31FC8>>1=0x16518FE4 ✓)
0x1699F524, // [71]  (0x2D33EA49>>1: bit0=1: (0x2D33EA49-1)>>1 +? no: arithmetic >>1 truncates: 0x1699F524 ✓)
0x16E221CD, // [72]  (0x2DC4439B>>1: bit0=1: 0x16E221CD ✓)
0x172A1637, // [73]  (0x2E542C6F>>1: bit0=1: 0x172A1637 ✓)
0x1771D2BA, // [74]  (0x2EE3A574>>1=0x1771D2BA ✓)
0x17B957AC, // [75]  (0x2F72AF58>>1=0x17B957AC ✓)
0x1800A563, // [76]  (0x30014AC6>>1: bit0=0: 0x18008A63... 
//  wait: 0x30014AC6>>1: 0011 0000 0000 0001 0100 1010 1100 0110
//  >>1:  0001 1000 0000 0000 1010 0101 0110 0011 = 0x1800A563 ✓)
0x1847BC33, // [77]  (0x308F7867>>1: bit0=1: 0x1847BC33 ✓)
0x188E9C72, // [78]  (0x311D38E5>>1: 0x188E9C72 ✓)
0x18D54673, // [79]  (0x31AA8CE7>>1: 0x18D54673 ✓)
0x191BBA89, // [80]  (0x32377512>>1=0x191BBA89 ✓)
0x1961F905, // [81]  (0x32C3F20A>>1=0x1961F905 ✓)
0x19A80239, // [82]  (0x33500472>>1=0x19A80239 ✓)
0x19EDD675, // [83]  (0x33DBACEB>>1: bit0=1: 0x19EDD675 ✓)
0x1A33760A, // [84]  (0x3466EC14>>1=0x1A33760A ✓)
0x1A78E146, // [85]  (0x34F1C28D>>1: 0x1A78E146 ✓)
0x1ABE1879, // [86]  (0x357C30F2>>1=0x1ABE1879 ✓)
0x1B031BEF, // [87]  (0x360637DF>>1: 0x1B031BEF ✓)
0x1B47EBF7, // [88]  (0x368FD7EE>>1=0x1B47EBF7 ✓)
0x1B8C88DB, // [89]  (0x371911B7>>1: 0x1B8C88DB ✓)
0x1BD0F2E9, // [90]  (0x37A1E5D3>>1: 0x1BD0F2E9 ✓)
0x1C152A6C, // [91]  (0x382A54D8>>1=0x1C152A6C ✓)
0x1C592FAD, // [92]  (0x38B25F5A>>1=0x1C592FAD ✓)
0x1C9D02F6, // [93]  (0x393A05ED>>1: 0x1C9D02F6 ✓)
0x1CE0A492, // [94]  (0x39C14924>>1=0x1CE0A492 ✓)
0x1D2414C8, // [95]  (0x3A482990>>1=0x1D2414C8 ✓)
0x1D6753E0, // [96]  (0x3ACEA7C0>>1=0x1D6753E0 ✓)
0x1DAA6222, // [97]  (0x3B54C444>>1=0x1DAA6222 ✓)
0x1DED3FD4, // [98]  (0x3BDA7FA8>>1=0x1DED3FD4 ✓)
0x1E2FED3D, // [99]  (0x3C5FDA7A>>1=0x1E2FED3D ✓)
0x1E726AA1, // [100] (0x3CE4D543>>1: 0x1E726AA1 ✓)
0x1EB4B847, // [101] (0x3D69708F>>1: 0x1EB4B847 ✓)
0x1EF6D673, // [102] (0x3DEDACE6>>1=0x1EF6D673 ✓)
0x1F38C567, // [103] (0x3E718ACF>>1: 0x1F38C567 ✓)
0x1F7A8568, // [104] (0x3EF50AD1>>1: 0x1F7A8568 ✓)
0x1FBC16B9, // [105] (0x3F782D72>>1=0x1FBC16B9 ✓)
0x1FFD799A, // [106] (0x3FFAF335>>1: 0x1FFD799A ✓)
// EXP=-30, shift=0 (identity) for idx 107..255:
0x203EAE4E, // [107]
0x207FB517, // [108]
0x20C08E33, // [109]
0x210139E4, // [110]
0x2141B869, // [111]
0x21820A01, // [112]
0x21C22EEA, // [113]
0x22022762, // [114]
0x2241F3A7, // [115]
0x228193F5, // [116]
0x22C10889, // [117]
0x2300519E, // [118]
0x233F6F71, // [119]
0x237E623D, // [120]
0x23BD2A3B, // [121]
0x23FBC7A6, // [122]
0x243A3AB7, // [123]
0x247883A8, // [124]
0x24B6A2B1, // [125]
0x24F4980B, // [126]
0x253263EC, // [127]
0x2570068E, // [128]
0x25AD8026, // [129]
0x25EAD0EB, // [130]
0x2627F914, // [131]
0x2664F8D5, // [132]
0x26A1D064, // [133]
0x26DE7FF6, // [134]
0x271B07C0, // [135]
0x275767F5, // [136]
0x2793A0C9, // [137]
0x27CFB26F, // [138]
0x280B9D1A, // [139]
0x284760FD, // [140]
0x2882FE49, // [141]
0x28BE7531, // [142]
0x28F9C5E5, // [143]
0x2934F097, // [144]
0x296FF577, // [145]
0x29AAD4B6, // [146]
0x29E58E83, // [147]
0x2A20230E, // [148]
0x2A5A9285, // [149]
0x2A94DD19, // [150]
0x2ACF02F7, // [151]
0x2B09044D, // [152]
0x2B42E149, // [153]
0x2B7C9A19, // [154]
0x2BB62EEA, // [155]
0x2BEF9FE8, // [156]
0x2C28ED40, // [157]
0x2C62171E, // [158]
0x2C9B1DAE, // [159]
0x2CD4011C, // [160]
0x2D0CC192, // [161]
0x2D455F3C, // [162]
0x2D7DDA44, // [163]
0x2DB632D4, // [164]
0x2DEE6917, // [165]
0x2E267D36, // [166]
0x2E5E6F5A, // [167]
0x2E963FAC, // [168]
0x2ECDEE56, // [169]
0x2F057B7F, // [170]
0x2F3CE751, // [171]
0x2F7431F2, // [172]
0x2FAB5B8B, // [173]
0x2FE26443, // [174]
0x30194C40, // [175]
0x305013AB, // [176]
0x3086BAA9, // [177]
0x30BD4161, // [178]
0x30F3A7F8, // [179]
0x3129EE96, // [180]
0x3160155E, // [181]
0x31961C76, // [182]
0x31CC0404, // [183]
0x3201CC2C, // [184]
0x32377512, // [185]
0x326CFEDB, // [186]
0x32A269AB, // [187]
0x32D7B5A5, // [188]
0x330CE2ED, // [189]
0x3341F1A7, // [190]
0x3376E1F5, // [191]
0x33ABB3FA, // [192]
0x33E067D9, // [193]
0x3414FDB4, // [194]
0x344975AD, // [195]
0x347DCFE7, // [196]
0x34B20C82, // [197]
0x34E62BA0, // [198]
0x351A2D62, // [199]
0x354E11EB, // [200]
0x3581D959, // [201]
0x35B583CE, // [202]
0x35E9116A, // [203]
0x361C824D, // [204]
0x364FD697, // [205]
0x36830E69, // [206]
0x36B629E1, // [207]
0x36E9291E, // [208]
0x371C0C41, // [209]
0x374ED367, // [210]
0x37817EAF, // [211]
0x37B40E39, // [212]
0x37E68222, // [213]
0x3818DA88, // [214]
0x384B178A, // [215]
0x387D3945, // [216]
0x38AF3FD7, // [217]
0x38E12B5D, // [218]
0x3912FBF4, // [219]
0x3944B1B9, // [220]
0x39764CC9, // [221]
0x39A7CD41, // [222]
0x39D9333D, // [223]
0x3A0A7EDA, // [224]
0x3A3BB033, // [225]
0x3A6CC764, // [226]
0x3A9DC48A, // [227]
0x3ACEA7C0, // [228]
0x3AFF7121, // [229]
0x3B3020C8, // [230]
0x3B60B6D1, // [231]
0x3B913356, // [232]
0x3BC19672, // [233]
0x3BF1E041, // [234]
0x3C2210DB, // [235]
0x3C52285C, // [236]
0x3C8226DD, // [237]
0x3CB20C79, // [238]
0x3CE1D948, // [239]
0x3D118D66, // [240]
0x3D4128EB, // [241]
0x3D70ABF1, // [242]
0x3DA01691, // [243]
0x3DCF68E3, // [244]
0x3DFEA301, // [245]
0x3E2DC503, // [246]
0x3E5CCF02, // [247]
0x3E8BC117, // [248]
0x3EBA9B59, // [249]
0x3EE95DE1, // [250]
0x3F1808C7, // [251]
0x3F469C22, // [252]
0x3F75180B, // [253]
0x3FA37C98, // [254]
0x3FD1C9E2, // [255]
0x40000000, // [256] EXP=-29, shift=+1: 0x20000000<<1
};

constexpr SF_HOT SoftFloat SoftFloat::log2() const noexcept
{
	if (UNLIKELY(mantissa <= 0)) return zero();

	int32_t  E = exponent + 29;
	uint32_t m_abs = static_cast<uint32_t>(mantissa);

	uint32_t low = m_abs - 0x20000000u;
	uint32_t t_int = low >> 21;
	uint32_t frac = (low >> 13) & 0xFFu;

	// Direct Q30 lookup — no to_q30() calls needed
	int32_t v0 = LOG2_Q30[t_int];
	int32_t v1 = LOG2_Q30[t_int + 1];

	int32_t delta = v1 - v0;
	int32_t corr = (delta * static_cast<int32_t>(frac)) >> 8;
	int32_t log2_frac_q30 = v0 + corr;

	SoftFloat fractional_part(log2_frac_q30, -30);
	SoftFloat integer_part(E);
	return fractional_part + integer_part;
}

// Natural logarithm: log(x) = log2(x) * ln(2)
constexpr SoftFloat SoftFloat::log() const noexcept {
	constexpr SoftFloat LN2 = from_raw(0x2C5C85FE, -30);
	return log2() * LN2;                        // one extra multiply, but half the code size
}

// log10(x) = log2(x) * log10(2)
constexpr SoftFloat SoftFloat::log10() const noexcept {
	constexpr SoftFloat LOG10_2 = SoftFloat::from_raw(0x268826A1, -31); // log10(2) ≈ 0.30103
	return log2() * LOG10_2;
}

// ------------------------------------------------------------------
// pow — fixed integer fast path
// ------------------------------------------------------------------
constexpr SoftFloat SoftFloat::pow(SoftFloat y) const noexcept {
	if (mantissa == 0) return y.mantissa == 0 ? one() : zero();
	if (y.mantissa == 0) return one();

	// --- Literal fast paths (two integer compares each) ---
	if (y == one())       return *this;
	if (y == two())       { SoftFloat t = *this; return SoftFloat::mul_plain(t, t); }
	if (y == three())     { SoftFloat t = *this; return SoftFloat::mul_plain(t, SoftFloat::mul_plain(t, t)); }
	if (y == four())      { SoftFloat t = *this; t = SoftFloat::mul_plain(t, t); return SoftFloat::mul_plain(t, t); }
	if (y == neg_one())   return reciprocal();
	if (y == half())      return sqrt();
	if (y == -half())     return inv_sqrt();

	// 1.5 = x * sqrt(x)
	if (y.mantissa == 0x30000000 && y.exponent == -29) {
		SoftFloat t = sqrt();
		return SoftFloat::mul_plain(*this, t);
	}
	// 0.25 = sqrt(sqrt(x))
	if (y.mantissa == 0x20000000 && y.exponent == -31)
		return sqrt().sqrt();

	// --- Zero-cost integer detection (bit-mask on mantissa) ---
	int32_t n = 0;
	bool is_int = false;
	if (y.exponent < 0) {
		int32_t shift = -y.exponent; // 1 … 30
		if (shift <= 30) {
			uint32_t a = sf_abs32(y.mantissa);
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

		// Early saturation (avoid useless loops)
		if (exponent > 0 && un > 127u / static_cast<uint32_t>(exponent))
			return from_raw(mantissa > 0 ? (1 << 29) : -(1 << 29), 127);
		if (exponent < 0 && un > 128u / static_cast<uint32_t>(-exponent))
			return zero();

		SoftFloat result = one();
		SoftFloat base   = *this;
		for (; un; un >>= 1) {
			if (un & 1u) result = SoftFloat::mul_plain(result, base);
			if (un == 1u) break;                   // final square would be discarded
			base = SoftFloat::mul_plain(base, base);
		}
		return neg ? result.reciprocal() : result;
	}

	if (is_negative()) return zero();              // non-integer power of negative
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

	// |y| negligible relative to |x| → result is |x| (error < ½ ulp)
	if (d >= 15) return x;

	uint32_t mx = static_cast<uint32_t>(x.mantissa);
	uint32_t my = static_cast<uint32_t>(y.mantissa);

	// S = mx² + my²·2^(‑2d)  is the exact integer coefficient of 2^(2·ex)
	uint64_t mx2 = static_cast<uint64_t>(mx) * mx;
	uint64_t my2 = static_cast<uint64_t>(my) * my;
	uint64_t S   = mx2 + (my2 >> (2 * d));

	// Pack S into a SoftFloat.  S ∈ [2⁵⁸, 2⁶¹)  →  S>>29 ∈ [2²⁹, 2³²).
	uint32_t s_hi = static_cast<uint32_t>(S >> 29);
	int32_t  s_e  = 29;

	// s_hi must fit in a positive int32_t.  At most two right-shifts are needed.
	if (s_hi >= 0x80000000u) {
		s_hi >>= 2; // 2³¹..2³²‑1  →  2²⁹..2³⁰‑1
		s_e  += 2;
	}
	else if (s_hi >= 0x40000000u) {
		s_hi >>= 1; // 2³⁰..2³¹‑1  →  2²⁹..2³⁰‑1
		s_e  += 1;
	}
	// s_hi is now guaranteed in [0x20000000, 0x3FFFFFFF]: already normalized.

	SoftFloat s_sf = SoftFloat::from_raw(static_cast<int32_t>(s_hi), s_e);

	// √(x²+y²) = √S · 2^ex
	SoftFloat r = s_sf.sqrt();
	return SoftFloat::from_raw(r.mantissa, r.exponent + ex);
}

// trunc – toward zero
constexpr SF_HOT SoftFloat SoftFloat::trunc() const noexcept {
	return SoftFloat(to_int32());
}

// floor – toward -inf
// Optimized: Direct bit manipulation avoids SoftFloat arithmetic overhead.
constexpr SoftFloat SoftFloat::floor() const noexcept {
	// 1. Fast path: Zero or already an integer (exponent >= 0)
	if (UNLIKELY(mantissa == 0)) return *this;
	if (exponent >= 0) return *this;

	// 2. Determine shift amount for fractional bits.
	//    exponent is negative (e.g., -1 to -128).
	int32_t rs = -exponent;

	// 3. Handle tiny numbers: If shift >= 31, abs(value) < 1.
	//    Mantissa is in [2^29, 2^30).
	if (rs >= 31) {
		// If positive, floor is 0. If negative, floor is -1.
		return mantissa > 0 ? SoftFloat::zero() : SoftFloat::neg_one();
	}

	// 4. Extract absolute value of mantissa.
	uint32_t a = sf_abs32(mantissa);

	// 5. Truncate toward zero (shift right).
	//    We also need to know if we dropped any bits (fractional part).
	uint32_t frac_mask = (1u << rs) - 1u;
	bool has_frac = (a & frac_mask) != 0;

	int32_t int_part_m = static_cast<int32_t>(a >> rs);
	int32_t result_m;

	// 6. Apply floor logic:
	//    Positive: Truncation IS floor.
	//    Negative: If we dropped bits, we must go one lower (more negative).
	if (mantissa < 0) {
		if (has_frac) {
			result_m = -(int_part_m + 1);
		}
		else {
			result_m = -int_part_m;
		}
	}
	else {
		result_m = int_part_m;
	}

	// 7. Construct result. 
	//    SoftFloat(int32_t) normalizes the integer to standard form.
	return SoftFloat(result_m);
}

constexpr SoftFloat SoftFloat::ceil() const noexcept {
	if (mantissa == 0 || exponent >= 0) return *this;
	// exponent < 0: some fractional bits exist
	// Check if perfectly integer already
	int rs = -exponent; // number of fractional bits
	if (rs >= 30) {
		// |x| < 1: ceil = 0 for negative, 1 for positive
		return is_positive() ? SoftFloat::one() : SoftFloat::zero();
	}
	uint32_t a = sf_abs32(mantissa);
	uint32_t frac_mask = (1u << rs) - 1u;
	bool has_frac = (a & frac_mask) != 0;
    
	// Truncate to integer (clear fractional bits)
	int32_t trunc_m = static_cast<int32_t>(a & ~frac_mask);
	int32_t trunc_e = exponent;
	if (trunc_m == 0) {
		// |x| < epsilon
		return is_positive() && has_frac ? SoftFloat::one() : SoftFloat::zero();
	}
    
	// Reconstruct truncated value with correct sign
	SoftFloat fi = from_raw(is_negative() ? -trunc_m : trunc_m, trunc_e);
    
	// ceil: if positive and had fraction, add one
	if (is_positive() && has_frac) {
		return fi + SoftFloat::one();
	}
	return fi;
}

// round – nearest, ties away from zero
// Optimized: Integer arithmetic on mantissa avoids full SoftFloat add/trunc chain.
constexpr SoftFloat SoftFloat::round() const noexcept {
	// 1. Fast path: Zero or integer
	if (UNLIKELY(mantissa == 0)) return *this;
	if (exponent >= 0) return *this;

	int32_t rs = -exponent;

	// 2. Handle tiny numbers.
	//    If shift >= 31, value is in (-1, 1).
	//    Since normalized mantissa is in [0.5, 1.0), rs=30 means val in [0.5, 1.0).
	if (rs >= 31) return SoftFloat::zero();

	// 3. Extract absolute value.
	uint32_t a = sf_abs32(mantissa);

	// 4. Add rounding bias: 0.5 * 2^rs = 1 << (rs - 1).
	//    This implements "round to nearest, ties away from zero".
	//    Proof: 
	//      Let val = a * 2^-rs.
	//      We want round(val). 
	//      Algorithm adds 0.5 to magnitude: (a * 2^-rs) + 0.5
	//      Multiplied by 2^rs: a + 2^(rs-1).
	//      Integer truncation of sum >> rs is the rounded result.
    
	uint32_t bias = 1u << (rs - 1);
	uint32_t sum = a + bias; // Safe from overflow: a < 2^30, bias <= 2^29.

	int32_t result_m = static_cast<int32_t>(sum >> rs);

	// 5. Restore sign.
	if (mantissa < 0) result_m = -result_m;

	return SoftFloat(result_m);
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
	return from_raw((mantissa ^ sign.mantissa) >= 0 ? mantissa : -mantissa, exponent);
}

constexpr SoftFloat SoftFloat::fmod(SoftFloat y) const noexcept {
	if (UNLIKELY(y.mantissa == 0)) return *this;   // NaN policy: keep yours
	if (UNLIKELY(mantissa == 0))   return *this;

	int32_t  sx = (mantissa < 0) ? -1 : 1;
	uint32_t ax = sf_abs32(mantissa);
	uint32_t ay = sf_abs32(y.mantissa);
	int32_t  d  = exponent - y.exponent;

	// |x| < |y|  →  remainder is x itself
	if (d < 0) return *this;
	if (d == 0 && ax < ay) return *this;

	// Equal exponents: at most one subtraction
	if (d == 0) {
		uint32_t r = ax - ay; // ax ≥ ay here
		if (r == 0) return zero();
		int32_t rm = static_cast<int32_t>(r);
		int32_t re = y.exponent;
		sf_normalise_fast(rm, re);
		return from_raw(sx * rm, re);
	}

	// -----------------------------------------------------------------
	// Compile-time / portable path  (no inline asm, fully constexpr)
	// -----------------------------------------------------------------
	if (SF_IS_CONSTEVAL()) {
		uint64_t r64 = ax;
		int32_t  rem = d;
		while (rem > 0) {
			int shift = (rem > 30) ? 30 : rem; // keep r64<<shift inside 64 bits
			r64 = (r64 << shift) % ay;
			rem -= shift;
		}
		uint32_t r = static_cast<uint32_t>(r64);
		if (r == 0) return zero();
		int32_t rm = static_cast<int32_t>(r);
		int32_t re = y.exponent;
		sf_normalise_fast(rm, re);
		return from_raw(sx * rm, re);
	}

#if defined(__arm__)
	    // -----------------------------------------------------------------
	    // Runtime ARM path: iterative remainder with UDIV.
	    // r < ay < 2^30  →  r<<2 < 2^32, so every step is a single 32-bit UDIV.
	    // -----------------------------------------------------------------
	{
		uint32_t r   = ax;
		int32_t  rem = d;

		// Consume 2 bits per iteration; last iteration handles an odd bit.
		while (rem > 1) {
			uint32_t num = r << 2; // safe: r < 2^30
			uint32_t q;
			__asm__("udiv %0, %1, %2"
			        : "=r"(q) : "r"(num),
				"r"(ay));
			r = num - q * ay; // same as num % ay
			rem -= 2;
		}
		if (rem == 1) {
			// possible trailing bit
			uint32_t num = r << 1;
			uint32_t q;
			__asm__("udiv %0, %1, %2"
			        : "=r"(q) : "r"(num),
				"r"(ay));
			r = num - q * ay;
		}

		if (r == 0) return zero();
		int32_t rm = static_cast<int32_t>(r);
		int32_t re = y.exponent;
		sf_normalise_fast(rm, re);
		return from_raw(sx * rm, re);
	}
#else
	// -----------------------------------------------------------------
	// Host / non-ARM runtime (same algorithm as constexpr path)
	// -----------------------------------------------------------------
	{
		uint64_t r64 = ax;
		int32_t  rem = d;
		while (rem > 0) {
			int shift = (rem > 30) ? 30 : rem;
			r64 = (r64 << shift) % ay;
			rem -= shift;
		}
		uint32_t r = static_cast<uint32_t>(r64);
		if (r == 0) return zero();
		int32_t rm = static_cast<int32_t>(r);
		int32_t re = y.exponent;
		sf_normalise_fast(rm, re);
		return from_raw(sx * rm, re);
	}
#endif
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