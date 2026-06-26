package field52simd

import "math/big"

// Batched field ops over Fe8 that are cheap enough to stay in pure Go (shared
// by both backends): add, subtract, and modular inverse. Only Mul/Sqr are the
// backend-selected kernel. All keep the invariant that the IFMA Mul/Sqr need:
// every limb < 2^52 (limb 4 < 2^48 + tiny), value < 2^256 + tiny.

// negBias[k] = 4 * (canonical 52-bit limb k of p). Used by SubBatch to compute
// a - b as a + (4p - b) so every limb stays non-negative without a borrow
// (4*p_limb[k] >= any input limb). Value stays a + 4p < 5p < 2^259.
var negBias = func() [5]uint64 {
	t := new(big.Int).Set(P)
	m := new(big.Int).SetUint64(mask52)
	var b [5]uint64
	for k := 0; k < 5; k++ {
		b[k] = new(big.Int).And(t, m).Uint64() * 4
		t.Rsh(t, 52)
	}
	return b
}()

// foldLane folds a single lane's overflow above 2^256 (limb 4 >> 48) back via
// c = 2^32+977, restoring the limb<2^52 / value<2^256+tiny invariant. Used after
// add/sub, where the incoming value is < ~2^259 so one fold suffices.
func foldLane(z *Fe8, l int) {
	mtop := z[4][l] >> 48
	z[4][l] &= (1 << 48) - 1
	v := z[0][l] + mtop*c52 // mtop < 2^4, c52 < 2^33 -> < 2^52
	z[0][l] = v & mask52
	carry := v >> 52
	for k := 1; k < 5 && carry != 0; k++ {
		v = z[k][l] + carry
		z[k][l] = v & mask52
		carry = v >> 52
	}
}

// IsZero reports whether the element is congruent to 0 mod p. (Canonicalizes
// via Bytes; only used on the rare degenerate-detection path in advance.)
func (z *Fe) IsZero() bool {
	return z.Bytes() == [32]byte{}
}

// AddBatch sets out[l] = a[l] + b[l] (mod p, denormalized) for all lanes.
func AddBatch(out, a, b *Fe8) {
	for l := 0; l < Lanes; l++ {
		var carry uint64
		for k := 0; k < 5; k++ {
			v := a[k][l] + b[k][l] + carry
			out[k][l] = v & mask52
			carry = v >> 52
		}
		foldLane(out, l)
	}
}

// SubBatch sets out[l] = a[l] - b[l] (mod p, denormalized) for all lanes.
func SubBatch(out, a, b *Fe8) {
	for l := 0; l < Lanes; l++ {
		var carry uint64
		for k := 0; k < 5; k++ {
			// a + 4p - b: a + negBias >= b so the subtraction never underflows.
			v := a[k][l] + negBias[k] + carry - b[k][l]
			out[k][l] = v & mask52
			carry = v >> 52
		}
		foldLane(out, l)
	}
}

// InverseFe8 is backend-selected: the IFMA build runs the whole Fermat chain in
// one cgo call (batch_ifma.go); the pure-Go build (batch_purego.go) chains
// MulBatch/SqrBatch.
