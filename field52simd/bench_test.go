package field52simd

import (
	"testing"

	"github.com/btcsuite/btcd/btcec/v2"
)

// BenchmarkMulBatch measures one 8-lane field multiply (the IFMA kernel under
// -tags avx512ifma, the pure-Go fallback otherwise). Divide ns/op by 8 for the
// per-field-multiply cost.
func BenchmarkMulBatch(b *testing.B) {
	var x, y, out Fe8
	for k := 0; k < 5; k++ {
		for l := 0; l < Lanes; l++ {
			x[k][l] = 0x000fffffffffffff - uint64(l)
			y[k][l] = 0x000ffffffffffff0 + uint64(k)
		}
	}
	x[4] = [Lanes]uint64{}
	y[4] = [Lanes]uint64{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MulBatch(&out, &x, &y)
	}
}

// BenchmarkBtcecMul is the scalar baseline: 8 secp256k1 field multiplies via
// btcec.FieldVal (radix 2^26), matching one MulBatch worth of work.
func BenchmarkBtcecMul(b *testing.B) {
	var a, c btcec.FieldVal
	a.SetInt(1).MulInt(7)
	c.SetInt(1).MulInt(9)
	for k := 0; k < 5; k++ {
		a.Add(&c)
	}
	a.Normalize()
	c.Normalize()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for l := 0; l < Lanes; l++ {
			var r btcec.FieldVal
			r.Mul2(&a, &c)
		}
	}
}
