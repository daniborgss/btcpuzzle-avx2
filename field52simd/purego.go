package field52simd

import (
	"encoding/binary"
	"math/big"
	"math/bits"
)

// pLimbs is p in radix 2^52 (canonical), used by the conditional subtraction in
// reduceCanonical. Matches libsecp256k1's field_5x52 prime constants.
var pLimbs = [5]uint64{
	0xFFFFEFFFFFC2F, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0x0FFFFFFFFFFFF,
}

const mask52 = (1 << 52) - 1

// Reduction constants from p = 2^256 - 2^32 - 977, i.e. 2^256 ≡ c (mod p).
//
//	c52  = c          = 2^32 + 977   — folds overflow at the 2^256 boundary.
//	cAdj = 16 * c     = 2^36 + 15632 — folds the high limbs, whose weight
//	                    starts at 2^260 (5*52); the extra 4 bits are the "* 16"
//	                    adjustment, exactly mirroring btcec field.go (~l.1032).
//
// In radix 2^52 both constants fit in a single 52-bit limb (unlike radix 2^26,
// where 16c needs two words), which is why the fold below is single-limb.
const (
	c52  = 4294968273  // 2^32 + 977
	cAdj = 68719492368 // 16 * c52 = 2^36 + 15632
)

// P is the secp256k1 field prime, 2^256 - 2^32 - 977.
var P = func() *big.Int {
	p := new(big.Int).Lsh(big.NewInt(1), 256)
	p.Sub(p, new(big.Int).Lsh(big.NewInt(1), 32))
	p.Sub(p, big.NewInt(977))
	return p
}()

// Fe is one secp256k1 field element in radix 2^52 (5 little-endian limbs).
// It is the single-lane scalar reference for the future 8-lane IFMA kernel.
type Fe struct{ n [5]uint64 }

// madd52lo emulates AVX-512 IFMA vpmadd52luq for one lane:
// acc += low 52 bits of (a*b), with a,b < 2^52.
func madd52lo(acc, a, b uint64) uint64 {
	_, lo := bits.Mul64(a, b)
	return acc + (lo & mask52)
}

// madd52hi emulates AVX-512 IFMA vpmadd52huq for one lane:
// acc += bits 52..103 of (a*b), i.e. (a*b) >> 52, with a,b < 2^52.
func madd52hi(acc, a, b uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	return acc + ((hi << 12) | (lo >> 52))
}

// toBig reconstructs the (possibly unreduced) integer value of the limbs.
func (z *Fe) toBig() *big.Int {
	r := new(big.Int)
	for k := 4; k >= 0; k-- {
		r.Lsh(r, 52)
		r.Add(r, new(big.Int).SetUint64(z.n[k]))
	}
	return r
}

// fromBig stores v (assumed in [0, 2^260)) into the 5 limbs, no reduction.
func (z *Fe) fromBig(v *big.Int) {
	t := new(big.Int).Set(v)
	m := new(big.Int).SetUint64(mask52)
	for k := 0; k < 5; k++ {
		z.n[k] = new(big.Int).And(t, m).Uint64()
		t.Rsh(t, 52)
	}
}

// SetBytes loads a 32-byte big-endian value (value < 2^256 fits in 5 limbs).
func (z *Fe) SetBytes(b *[32]byte) {
	z.fromBig(new(big.Int).SetBytes(b[:]))
}

// reduceCanonical reduces the limbs in place to the unique canonical form
// (value < p; limbs 0..3 < 2^52, limb 4 < 2^48), in pure limb arithmetic with
// no big.Int. Accepts any input whose limbs/value the pipeline produces
// (limbs up to a raw Add's < 2^53, value up to a sub chain's < ~2^259).
func (z *Fe) reduceCanonical() {
	// Carry-normalize (handles limbs > 2^52 from a raw Add).
	var carry uint64
	for k := 0; k < 5; k++ {
		v := z.n[k] + carry
		z.n[k] = v & mask52
		carry = v >> 52
	}
	// Fold any value >= 2^260 (16c), then iterate the 2^256 fold (c) until the
	// top limb is < 2^48. Both are tiny for our inputs (at most a couple steps).
	for carry != 0 {
		v := z.n[0] + carry*cAdj
		z.n[0] = v & mask52
		c2 := v >> 52
		for k := 1; k < 5; k++ {
			v = z.n[k] + c2
			z.n[k] = v & mask52
			c2 = v >> 52
		}
		carry = c2
	}
	for z.n[4] >= (1 << 48) {
		mtop := z.n[4] >> 48
		z.n[4] &= (1 << 48) - 1
		v := z.n[0] + mtop*c52
		z.n[0] = v & mask52
		c2 := v >> 52
		for k := 1; k < 5; k++ {
			v = z.n[k] + c2
			z.n[k] = v & mask52
			c2 = v >> 52
		}
	}
	// Conditional subtract p: compute z - p with borrow; if it didn't borrow
	// (z >= p), keep the difference. value < 2^256 < 2p, so one step suffices.
	var t [5]uint64
	var borrow uint64
	for k := 0; k < 5; k++ {
		d := (z.n[k] | (1 << 52)) - pLimbs[k] - borrow
		t[k] = d & mask52
		borrow = 1 - ((d >> 52) & 1)
	}
	if borrow == 0 {
		z.n = t
	}
}

// Bytes returns the canonical (< p) value as 32 big-endian bytes.
func (z *Fe) Bytes() [32]byte {
	t := *z
	t.reduceCanonical()
	w0 := t.n[0] | (t.n[1] << 52)
	w1 := (t.n[1] >> 12) | (t.n[2] << 40)
	w2 := (t.n[2] >> 24) | (t.n[3] << 28)
	w3 := (t.n[3] >> 36) | (t.n[4] << 16)
	var out [32]byte
	binary.BigEndian.PutUint64(out[0:], w3)
	binary.BigEndian.PutUint64(out[8:], w2)
	binary.BigEndian.PutUint64(out[16:], w1)
	binary.BigEndian.PutUint64(out[24:], w0)
	return out
}

// Normalize reduces the element to its canonical representative mod p.
func (z *Fe) Normalize() {
	z.fromBig(new(big.Int).Mod(z.toBig(), P))
}

// Add sets z = x + y (limbs simply summed; magnitude grows, like btcec.Add).
func (z *Fe) Add(x, y *Fe) {
	for k := 0; k < 5; k++ {
		z.n[k] = x.n[k] + y.n[k]
	}
}

// Negate sets z ≡ -x (mod p).
func (z *Fe) Negate(x *Fe) {
	v := new(big.Int).Mod(x.toBig(), P)
	v.Sub(P, v)
	v.Mod(v, P)
	z.fromBig(v)
}

// reduceSolinas reduces a 10-limb radix-2^52 product (each column < 2^52 after
// the carry pass, value < 2^520) to 5 limbs congruent mod p. The result has
// magnitude 1 (limbs 0..3 < 2^52, limb 4 < 2^48 + 1) but may be denormalized
// (value can be slightly >= p); callers canonicalize via Normalize/Bytes.
//
// It is the limb-wise analog of btcec's mulInner reduction (field.go ~l.1042)
// and the exact sequence the AVX-512 IFMA kernel will implement with vpmadd52:
// every fold product (limb * constant) is split low/high via madd52lo/hi, both
// operands < 2^52.
//
//	Round 1: fold columns 5..9 (weight >= 2^260) with cAdj = 16c, leaving a
//	         small column 5 (r5).
//	Round 2: fold r5 (weight 2^260) with cAdj.
//	Round 3: fold the final overflow above 2^256 (limb 4 >> 48) with c (unscaled,
//	         because that overflow is in units of 2^256, not 2^260).
func reduceSolinas(t *[10]uint64) [5]uint64 {
	lo := func(h, c uint64) uint64 { return madd52lo(0, h, c) }
	hi := func(h, c uint64) uint64 { return madd52hi(0, h, c) }

	var r [5]uint64
	var m uint64

	// Round 1.
	m = t[0] + lo(t[5], cAdj)
	r[0] = m & mask52
	m = (m >> 52) + t[1] + hi(t[5], cAdj) + lo(t[6], cAdj)
	r[1] = m & mask52
	m = (m >> 52) + t[2] + hi(t[6], cAdj) + lo(t[7], cAdj)
	r[2] = m & mask52
	m = (m >> 52) + t[3] + hi(t[7], cAdj) + lo(t[8], cAdj)
	r[3] = m & mask52
	m = (m >> 52) + t[4] + hi(t[8], cAdj) + lo(t[9], cAdj)
	r[4] = m & mask52
	r5 := (m >> 52) + hi(t[9], cAdj) // column 5 (weight 2^260), < 2^38

	// Round 2.
	m = r[0] + lo(r5, cAdj)
	r[0] = m & mask52
	m = (m >> 52) + r[1] + hi(r5, cAdj)
	r[1] = m & mask52
	m = (m >> 52) + r[2]
	r[2] = m & mask52
	m = (m >> 52) + r[3]
	r[3] = m & mask52
	r[4] = (m >> 52) + r[4]

	// Round 3.
	mtop := r[4] >> 48
	r[4] &= (1 << 48) - 1
	m = r[0] + lo(mtop, c52)
	r[0] = m & mask52
	m = (m >> 52) + r[1] + hi(mtop, c52)
	r[1] = m & mask52
	m = (m >> 52) + r[2]
	r[2] = m & mask52
	m = (m >> 52) + r[3]
	r[3] = m & mask52
	r[4] = (m >> 52) + r[4]

	return r
}

// Mul sets z = x*y mod p.
//
// The 5x5 schoolbook emulates the IFMA kernel exactly: each partial product
// x[i]*y[j] is split into its low-52 (vpmadd52luq) and high-52 (vpmadd52huq)
// halves and accumulated into columns i+j and i+j+1, followed by a radix-2^52
// carry pass and the limb-wise Solinas reduction. The whole path is now
// big.Int-free and directly translatable to vpmadd52.
//
// Precondition: input limbs < 2^53 (the headroom an Add leaves); test inputs
// from SetBytes are < 2^52 (limb 4 < 2^48).
func (z *Fe) Mul(x, y *Fe) {
	var t [10]uint64
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			t[i+j] = madd52lo(t[i+j], x.n[i], y.n[j])
			t[i+j+1] = madd52hi(t[i+j+1], x.n[i], y.n[j])
		}
	}

	// Radix-2^52 carry normalization across the 10 product columns.
	var carry uint64
	for k := 0; k < 10; k++ {
		v := t[k] + carry
		t[k] = v & mask52
		carry = v >> 52
	}

	z.n = reduceSolinas(&t)
}

// Sqr sets z = x*x mod p, specialized: the 10 cross products a[i]*a[j] (i<j)
// are computed once and the whole column is doubled, then the 5 diagonal terms
// a[i]^2 are added undoubled — 15 multiplies instead of Mul's 25 (~40% fewer).
//
// IFMA note: the operand cannot be pre-doubled (2*a[j] is 53 bits and vpmadd52
// reads only 52), so the doubling is a per-column shift (one op per column),
// which is exactly how the kernel will do it.
//
// Precondition: input limbs < 2^52 so that 2*cross + diagonal stays < 2^57.
func (z *Fe) Sqr(x *Fe) {
	a := &x.n
	var t [10]uint64

	// Cross products a[i]*a[j], i<j, accumulated once (columns 1..8).
	for i := 0; i < 5; i++ {
		for j := i + 1; j < 5; j++ {
			t[i+j] = madd52lo(t[i+j], a[i], a[j])
			t[i+j+1] = madd52hi(t[i+j+1], a[i], a[j])
		}
	}
	// Double the cross contribution (each column < 2^54 -> < 2^55).
	for k := 0; k < 10; k++ {
		t[k] <<= 1
	}
	// Add the diagonal terms a[i]^2 (undoubled): low->col 2i, high->col 2i+1.
	for i := 0; i < 5; i++ {
		t[2*i] = madd52lo(t[2*i], a[i], a[i])
		t[2*i+1] = madd52hi(t[2*i+1], a[i], a[i])
	}

	// Radix-2^52 carry normalization across the 10 product columns.
	var carry uint64
	for k := 0; k < 10; k++ {
		v := t[k] + carry
		t[k] = v & mask52
		carry = v >> 52
	}

	z.n = reduceSolinas(&t)
}
