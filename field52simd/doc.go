// Package field52simd is the staging ground for an AVX-512 IFMA (vpmadd52)
// vectorized secp256k1 field multiply. It mirrors the swappable-backend layout
// of ripemd160simd/: a build-tag-selected SIMD kernel will expose the same API
// as the pure-Go reference here, and the differential test harness validates
// whatever backend is selected byte-for-byte against the trusted scalar
// implementation in github.com/btcsuite/btcd/btcec/v2 (secp256k1/v4 FieldVal).
//
// # Representation
//
// A field element is held in radix 2^52 as 5 little-endian limbs (limb 0 is the
// least significant). 5*52 = 260 >= 256, and a normalized value keeps limbs
// 0..3 in [0,2^52) and limb 4 in [0,2^48). The 12 spare bits in each 64-bit
// lane are the headroom IFMA needs to accumulate vpmadd52 partial products
// without carry overflow. The eventual SIMD type is the 8-lane SoA form
// ([5]__m512i); this single-lane Fe is the scalar reference it must match.
//
// # The prime and its reduction constant
//
// p = 2^256 - 2^32 - 977, so 2^256 ≡ (2^32 + 977) (mod p). The scalar btcec
// code reduces in radix 2^26 using the 977 constant (see field.go). The IFMA
// kernel will reduce in radix 2^52 with the same congruence; because 256 is not
// a multiple of 52, that fold straddles a limb boundary and is the single most
// bug-prone step — hence Camada 1 (this harness) exists before it is written.
//
// # Testing contract (Camada 1)
//
// Every operation (Mul, Sqr, Add, Negate, Normalize) is checked against
// btcec.FieldVal over: uniform-random inputs, an edge-case/KAT table
// (0, 1, p-1, carry-maximizing limbs), and algebraic properties
// (commutativity, identity, distributivity, x*x == Sqr(x)). A go-fuzz target
// feeds identical bytes to both implementations. When the C IFMA backend lands,
// it is dropped in behind this same harness with no test changes.
package field52simd
