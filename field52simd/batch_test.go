package field52simd

import (
	"math/big"
	"testing"
)

// These tests run under both backends. With no tag, MulBatch/SqrBatch are the
// pure-Go fallback (sanity on the pack/unpack plumbing). With -tags avx512ifma,
// they are the C kernel, validated lane-by-lane against the single-lane scalar
// reference (Mul/Sqr) — this is Camada 3 (lane isolation) plus kernel
// correctness in one. Comparison is via Bytes() so any equivalent denormalized
// limb form is accepted; the value mod p must match.

func packFromBytes(t *testing.T, raw *[Lanes][32]byte) Fe8 {
	t.Helper()
	var in [Lanes]Fe
	for l := 0; l < Lanes; l++ {
		in[l].SetBytes(&raw[l])
	}
	var soa Fe8
	PackLanes(&soa, &in)
	return soa
}

func TestMulBatchVsReference(t *testing.T) {
	for iter := 0; iter < 20000; iter++ {
		var ra, rb [Lanes][32]byte
		var want [Lanes][32]byte
		for l := 0; l < Lanes; l++ {
			ra[l] = randBytes(t)
			rb[l] = randBytes(t)
			var a, b, w Fe
			a.SetBytes(&ra[l])
			b.SetBytes(&rb[l])
			w.Mul(&a, &b)
			want[l] = w.Bytes()
		}
		sa := packFromBytes(t, &ra)
		sb := packFromBytes(t, &rb)
		var sout Fe8
		MulBatch(&sout, &sa, &sb)
		var got [Lanes]Fe
		UnpackLanes(&got, &sout)
		for l := 0; l < Lanes; l++ {
			if g := got[l].Bytes(); g != want[l] {
				t.Fatalf("MulBatch lane %d\n a   =%x\n b   =%x\n got =%x\n want=%x",
					l, ra[l], rb[l], g, want[l])
			}
		}
	}
}

func TestSqrBatchVsReference(t *testing.T) {
	for iter := 0; iter < 20000; iter++ {
		var ra [Lanes][32]byte
		var want [Lanes][32]byte
		for l := 0; l < Lanes; l++ {
			ra[l] = randBytes(t)
			var a, w Fe
			a.SetBytes(&ra[l])
			w.Sqr(&a)
			want[l] = w.Bytes()
		}
		sa := packFromBytes(t, &ra)
		var sout Fe8
		SqrBatch(&sout, &sa)
		var got [Lanes]Fe
		UnpackLanes(&got, &sout)
		for l := 0; l < Lanes; l++ {
			if g := got[l].Bytes(); g != want[l] {
				t.Fatalf("SqrBatch lane %d\n a   =%x\n got =%x\n want=%x", l, ra[l], g, want[l])
			}
		}
	}
}

// TestBatchLaneIsolation puts an adversarial value (carry-maximizing) in one
// lane and a trivial value in the rest, then confirms every lane is computed
// independently — a kernel that leaks carries across lanes fails here.
func TestBatchLaneIsolation(t *testing.T) {
	odd := edgeValues() // includes all-ones (carry-maximizing) and p-1
	for _, adv := range odd {
		for victim := 0; victim < Lanes; victim++ {
			var ra, rb [Lanes][32]byte
			for l := 0; l < Lanes; l++ {
				if l == victim {
					ra[l], rb[l] = adv, adv
				} else {
					ra[l] = bytesOf(big.NewInt(int64(l) + 2))
					rb[l] = bytesOf(big.NewInt(int64(l) + 3))
				}
			}
			sa := packFromBytes(t, &ra)
			sb := packFromBytes(t, &rb)
			var sout Fe8
			MulBatch(&sout, &sa, &sb)
			var got [Lanes]Fe
			UnpackLanes(&got, &sout)
			for l := 0; l < Lanes; l++ {
				var a, b, w Fe
				a.SetBytes(&ra[l])
				b.SetBytes(&rb[l])
				w.Mul(&a, &b)
				if got[l].Bytes() != w.Bytes() {
					t.Fatalf("lane %d (victim=%d) wrong: cross-lane contamination?", l, victim)
				}
			}
		}
	}
}
