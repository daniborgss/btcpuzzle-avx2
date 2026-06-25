package field52simd

import (
	"math/big"
	"testing"

	"github.com/btcsuite/btcd/btcec/v2"
)

// buildRadix26 splits a value < 2^256 into 10 normalized radix-2^26 words — the
// definitional btcec layout. This lets us drive fromRadix26 without reaching
// into FieldVal's unexported word array.
func buildRadix26(v *big.Int) [10]uint32 {
	t := new(big.Int).Set(v)
	m := big.NewInt(mask26)
	var w [10]uint32
	for i := 0; i < 10; i++ {
		w[i] = uint32(new(big.Int).And(t, m).Uint64())
		t.Rsh(t, 26)
	}
	return w
}

// ---- base change: radix 2^26 <-> radix 2^52 --------------------------------

func TestRadix26PackUnpack(t *testing.T) {
	vals := edgeValues()
	for i := 0; i < 50000; i++ {
		vals = append(vals, randBytes(t))
	}
	for _, b := range vals {
		v := new(big.Int).SetBytes(b[:])

		// fromRadix26 of the definitional words must equal SetBytes.
		w := buildRadix26(v)
		var fromR26 Fe
		fromR26.fromRadix26(&w)

		var fromBytes Fe
		fromBytes.SetBytes(&b)

		if fromR26.Bytes() != fromBytes.Bytes() {
			t.Fatalf("fromRadix26 != SetBytes\n v=%x\n r26=%x\n byt=%x",
				b, fromR26.Bytes(), fromBytes.Bytes())
		}

		// toRadix26 is a clean inverse: pack(unpack(fe)) == fe.
		got := fromBytes.toRadix26()
		var back Fe
		back.fromRadix26(&got)
		if back != fromBytes {
			t.Fatalf("toRadix26 round-trip changed limbs\n v=%x\n in =%v\n out=%v",
				b, fromBytes.n, back.n)
		}
	}
}

// TestRadix26AgainstBtcec ties the base change to btcec: a value's radix-2^26
// words (built per definition) must pack to the same canonical bytes that a
// real btcec.FieldVal produces for that value.
func TestRadix26AgainstBtcec(t *testing.T) {
	for i := 0; i < 50000; i++ {
		b := randBytes(t)
		v := new(big.Int).SetBytes(b[:])

		var fv btcec.FieldVal
		fv.SetBytes(&b)
		want := fvBytes(&fv)

		w := buildRadix26(v)
		var fe Fe
		fe.fromRadix26(&w)
		if got := fe.Bytes(); got != want {
			t.Fatalf("radix26 vs btcec\n v=%x\n got =%x\n want=%x", b, got, want)
		}
	}
}

// ---- SoA transpose: [8]Fe <-> Fe8 ------------------------------------------

func randFe(t *testing.T) Fe {
	t.Helper()
	b := randBytes(t)
	var z Fe
	z.SetBytes(&b)
	return z
}

func TestPackLanesRoundTrip(t *testing.T) {
	for iter := 0; iter < 20000; iter++ {
		var in [Lanes]Fe
		for l := range in {
			in[l] = randFe(t)
		}
		var soa Fe8
		PackLanes(&soa, &in)
		var back [Lanes]Fe
		UnpackLanes(&back, &soa)
		if back != in {
			t.Fatalf("PackLanes/UnpackLanes round-trip mismatch\n in  =%v\n back=%v", in, back)
		}
	}
}

// TestLaneIsolation is the embryonic Camada 3: it asserts the SoA slot mapping
// is exactly out[k][l] == in[l].n[k], and that rewriting a single lane in SoA
// form perturbs only that lane on the way back (no cross-lane contamination —
// the classic SIMD transpose / vpmadd52 bug).
func TestLaneIsolation(t *testing.T) {
	var in [Lanes]Fe
	for l := range in {
		// Make every limb of every lane distinguishable: l in the low byte,
		// k in the next, so a swapped index is immediately visible.
		for k := 0; k < 5; k++ {
			in[l].n[k] = uint64(l) | uint64(k)<<8 | 0x1000
		}
	}
	var soa Fe8
	PackLanes(&soa, &in)
	for l := 0; l < Lanes; l++ {
		for k := 0; k < 5; k++ {
			if soa[k][l] != in[l].n[k] {
				t.Fatalf("slot map wrong at k=%d l=%d: got %#x want %#x",
					k, l, soa[k][l], in[l].n[k])
			}
		}
	}

	// Perturb one lane in SoA form; only that lane must change after unpack.
	const victim = 3
	for k := 0; k < 5; k++ {
		soa[k][victim] ^= 0xdeadbeef
	}
	var back [Lanes]Fe
	UnpackLanes(&back, &soa)
	for l := 0; l < Lanes; l++ {
		changed := back[l] != in[l]
		if l == victim && !changed {
			t.Fatalf("victim lane %d did not change", l)
		}
		if l != victim && changed {
			t.Fatalf("cross-lane contamination: lane %d changed when only %d was touched", l, victim)
		}
	}
}

// TestBoundaryEndToEnd exercises the full production boundary: 8 btcec values
// -> [8]Fe (byte bridge) -> SoA -> back -> bytes, and confirms each lane equals
// what btcec says for that value.
func TestBoundaryEndToEnd(t *testing.T) {
	for iter := 0; iter < 5000; iter++ {
		var raw [Lanes][32]byte
		var want [Lanes][32]byte
		var in [Lanes]Fe
		for l := 0; l < Lanes; l++ {
			raw[l] = randBytes(t)
			var fv btcec.FieldVal
			fv.SetBytes(&raw[l])
			want[l] = fvBytes(&fv)
			in[l].SetBytes(&raw[l])
		}
		var soa Fe8
		PackLanes(&soa, &in)
		var back [Lanes]Fe
		UnpackLanes(&back, &soa)
		for l := 0; l < Lanes; l++ {
			if got := back[l].Bytes(); got != want[l] {
				t.Fatalf("lane %d boundary mismatch\n got =%x\n want=%x", l, got, want[l])
			}
		}
	}
}
