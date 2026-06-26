package field52simd

import (
	"math/big"
	"testing"
)

func randFe8(t *testing.T) (Fe8, [Lanes][32]byte) {
	t.Helper()
	var raw [Lanes][32]byte
	var in [Lanes]Fe
	for l := 0; l < Lanes; l++ {
		raw[l] = randBytes(t)
		in[l].SetBytes(&raw[l])
	}
	var soa Fe8
	PackLanes(&soa, &in)
	return soa, raw
}

func TestAddBatch(t *testing.T) {
	for iter := 0; iter < 20000; iter++ {
		sa, ra := randFe8(t)
		sb, rb := randFe8(t)
		var sout Fe8
		AddBatch(&sout, &sa, &sb)
		var got [Lanes]Fe
		UnpackLanes(&got, &sout)
		for l := 0; l < Lanes; l++ {
			va := new(big.Int).SetBytes(ra[l][:])
			vb := new(big.Int).SetBytes(rb[l][:])
			want := bytesOf(new(big.Int).Add(va, vb))
			if got[l].Bytes() != want {
				t.Fatalf("AddBatch lane %d: a=%x b=%x got=%x want=%x", l, ra[l], rb[l], got[l].Bytes(), want)
			}
		}
	}
}

func TestSubBatch(t *testing.T) {
	for iter := 0; iter < 20000; iter++ {
		sa, ra := randFe8(t)
		sb, rb := randFe8(t)
		var sout Fe8
		SubBatch(&sout, &sa, &sb)
		var got [Lanes]Fe
		UnpackLanes(&got, &sout)
		for l := 0; l < Lanes; l++ {
			va := new(big.Int).SetBytes(ra[l][:])
			vb := new(big.Int).SetBytes(rb[l][:])
			want := bytesOf(new(big.Int).Sub(va, vb))
			if got[l].Bytes() != want {
				t.Fatalf("SubBatch lane %d: a=%x b=%x got=%x want=%x", l, ra[l], rb[l], got[l].Bytes(), want)
			}
		}
	}
}

// TestAddSubChain stresses the invariant under chained ops (the pattern advance
// uses): repeatedly add then subtract should return to the original value, and
// feeding denormalized results into MulBatch must still be correct.
func TestAddSubChain(t *testing.T) {
	for iter := 0; iter < 5000; iter++ {
		sa, ra := randFe8(t)
		sb, _ := randFe8(t)
		var s Fe8
		AddBatch(&s, &sa, &sb) // a+b
		SubBatch(&s, &s, &sb)  // (a+b)-b == a
		var got [Lanes]Fe
		UnpackLanes(&got, &s)
		for l := 0; l < Lanes; l++ {
			want := bytesOf(new(big.Int).SetBytes(ra[l][:]))
			if got[l].Bytes() != want {
				t.Fatalf("AddSub chain lane %d: got=%x want=%x", l, got[l].Bytes(), want)
			}
		}
		// Denormalized result fed into MulBatch must still be correct:
		// (a+b-b) * a == a*a mod p.
		var prod Fe8
		MulBatch(&prod, &s, &sa)
		var pgot [Lanes]Fe
		UnpackLanes(&pgot, &prod)
		for l := 0; l < Lanes; l++ {
			va := new(big.Int).SetBytes(ra[l][:])
			want := bytesOf(new(big.Int).Mul(va, va))
			if pgot[l].Bytes() != want {
				t.Fatalf("denormalized MulBatch lane %d: got=%x want=%x", l, pgot[l].Bytes(), want)
			}
		}
	}
}

func TestInverseBatch(t *testing.T) {
	var one Fe
	oneB := bytesOf(big.NewInt(1))
	one.SetBytes(&oneB)
	wantOne := one.Bytes()

	for iter := 0; iter < 5000; iter++ {
		sa, ra := randFe8(t)
		// Avoid zero lanes (inverse undefined); randBytes is ~never 0 but guard.
		skip := false
		for l := 0; l < Lanes; l++ {
			if new(big.Int).SetBytes(ra[l][:]).Sign() == 0 {
				skip = true
			}
		}
		if skip {
			continue
		}
		var sinv, prod Fe8
		InverseFe8(&sinv, &sa)
		MulBatch(&prod, &sa, &sinv) // a * a^-1 must be 1
		var got [Lanes]Fe
		UnpackLanes(&got, &prod)
		for l := 0; l < Lanes; l++ {
			if got[l].Bytes() != wantOne {
				t.Fatalf("InverseBatch lane %d: a=%x a*inv=%x (want 1)", l, ra[l], got[l].Bytes())
			}
		}
	}
}
