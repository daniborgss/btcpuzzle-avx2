package field52simd

// Lanes is the SIMD width of the future IFMA kernel: 8 × 64-bit lanes per
// 512-bit register.
const Lanes = 8

// Fe8 holds 8 field elements in the Structure-of-Arrays layout the IFMA kernel
// consumes: fe8[k][lane] is limb k of that lane's element. It is memory-
// identical to [5]__m512i (5 limbs, each a 512-bit vector of 8 lanes), so a C
// kernel can take *Fe8 via unsafe.Pointer with no repacking.
type Fe8 [5][Lanes]uint64

const mask26 = (1 << 26) - 1

// fromRadix26 packs btcec's radix-2^26 representation (10 little-endian 26-bit
// words) into this radix-2^52 element. Because 52 = 2*26, each 52-bit limb is
// exactly a pair of consecutive 26-bit words — the base change never splits a
// word, so it is just an OR of a shifted pair.
//
// Precondition: words normalized (each < 2^26, word 9 < 2^22), i.e. the source
// FieldVal had magnitude 1 / was Normalized.
func (z *Fe) fromRadix26(w *[10]uint32) {
	for i := 0; i < 5; i++ {
		z.n[i] = uint64(w[2*i]) | uint64(w[2*i+1])<<26
	}
}

// toRadix26 is the inverse: split each 52-bit limb into its low and high 26-bit
// words. Precondition: magnitude 1 (limbs < 2^52, limb 4 < 2^48).
func (z *Fe) toRadix26() [10]uint32 {
	var w [10]uint32
	for i := 0; i < 5; i++ {
		w[2*i] = uint32(z.n[i] & mask26)
		w[2*i+1] = uint32((z.n[i] >> 26) & mask26)
	}
	return w
}

// PackLanes transposes 8 elements from AoS ([8]Fe) into SoA (Fe8). This is the
// per-batch transpose the kernel needs before each vector multiply; the SIMD
// build does it with 5 vector loads + a register transpose, this scalar
// reference does it by definition. It is also the Camada-3 lane-isolation
// oracle: out[k][l] must equal in[l].n[k] and nothing else.
func PackLanes(out *Fe8, in *[Lanes]Fe) {
	for l := 0; l < Lanes; l++ {
		for k := 0; k < 5; k++ {
			out[k][l] = in[l].n[k]
		}
	}
}

// UnpackLanes transposes SoA (Fe8) back to AoS ([8]Fe) after the kernel runs.
func UnpackLanes(out *[Lanes]Fe, in *Fe8) {
	for l := 0; l < Lanes; l++ {
		for k := 0; k < 5; k++ {
			out[l].n[k] = in[k][l]
		}
	}
}
