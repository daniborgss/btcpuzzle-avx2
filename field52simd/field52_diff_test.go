package field52simd

import (
	"crypto/rand"
	"math/big"
	"testing"

	"github.com/btcsuite/btcd/btcec/v2"
)

// ---- bridges between the two implementations -------------------------------

// fvBytes normalizes a btcec.FieldVal (the oracle) and returns its 32 bytes.
func fvBytes(f *btcec.FieldVal) [32]byte {
	f.Normalize()
	return *f.Bytes()
}

func newFv(b *[32]byte) *btcec.FieldVal {
	var f btcec.FieldVal
	f.SetBytes(b)
	return &f
}

func newFe(b *[32]byte) *Fe {
	var z Fe
	z.SetBytes(b)
	return &z
}

// bytesOf renders a big.Int (taken mod p) as 32 big-endian bytes.
func bytesOf(v *big.Int) [32]byte {
	var b [32]byte
	new(big.Int).Mod(v, P).FillBytes(b[:])
	return b
}

func randBytes(t *testing.T) [32]byte {
	t.Helper()
	v, err := rand.Int(rand.Reader, P)
	if err != nil {
		t.Fatal(err)
	}
	return bytesOf(v)
}

// coerce maps an arbitrary byte slice (fuzz input) to a valid in-range element.
func coerce(bs []byte) [32]byte {
	return bytesOf(new(big.Int).SetBytes(bs))
}

// ---- per-operation differential checks (one input pair) --------------------

func checkMul(t *testing.T, ab, bb [32]byte) {
	t.Helper()
	fa, fb := newFv(&ab), newFv(&bb)
	fa.Mul(fb)
	want := fvBytes(fa)

	var z Fe
	z.Mul(newFe(&ab), newFe(&bb))
	if got := z.Bytes(); got != want {
		t.Fatalf("Mul mismatch\n a   =%x\n b   =%x\n got =%x\n want=%x", ab, bb, got, want)
	}
}

func checkSqr(t *testing.T, ab [32]byte) {
	t.Helper()
	fa := newFv(&ab)
	fa.Square()
	want := fvBytes(fa)

	var z Fe
	z.Sqr(newFe(&ab))
	if got := z.Bytes(); got != want {
		t.Fatalf("Sqr mismatch\n a   =%x\n got =%x\n want=%x", ab, got, want)
	}
}

func checkAdd(t *testing.T, ab, bb [32]byte) {
	t.Helper()
	fa, fb := newFv(&ab), newFv(&bb)
	fa.Add(fb)
	want := fvBytes(fa)

	var z Fe
	z.Add(newFe(&ab), newFe(&bb))
	if got := z.Bytes(); got != want {
		t.Fatalf("Add mismatch\n a   =%x\n b   =%x\n got =%x\n want=%x", ab, bb, got, want)
	}
}

func checkNegate(t *testing.T, ab [32]byte) {
	t.Helper()
	fa := newFv(&ab)
	fa.Normalize()
	var r btcec.FieldVal
	r.NegateVal(fa, 1)
	want := fvBytes(&r)

	var z Fe
	z.Negate(newFe(&ab))
	if got := z.Bytes(); got != want {
		t.Fatalf("Negate mismatch\n a   =%x\n got =%x\n want=%x", ab, got, want)
	}
}

// ---- uniform-random differential tests -------------------------------------

const randIters = 200000

func TestMulRandom(t *testing.T) {
	for i := 0; i < randIters; i++ {
		checkMul(t, randBytes(t), randBytes(t))
	}
}

func TestSqrRandom(t *testing.T) {
	for i := 0; i < randIters; i++ {
		checkSqr(t, randBytes(t))
	}
}

func TestAddRandom(t *testing.T) {
	for i := 0; i < randIters; i++ {
		checkAdd(t, randBytes(t), randBytes(t))
	}
}

func TestNegateRandom(t *testing.T) {
	for i := 0; i < randIters; i++ {
		checkNegate(t, randBytes(t))
	}
}

// ---- edge-case / KAT table -------------------------------------------------

// edgeValues spans the boundary conditions most likely to expose carry and
// reduction bugs: zero, small values, values just below p, the half prime, and
// limb-pattern values that maximize carry propagation in radix 2^52.
func edgeValues() [][32]byte {
	bi := func(v *big.Int) [32]byte { return bytesOf(v) }
	one := big.NewInt(1)
	pm1 := new(big.Int).Sub(P, one)
	pm2 := new(big.Int).Sub(P, big.NewInt(2))
	half := new(big.Int).Rsh(P, 1)
	// all limbs = 2^52-1 across limbs 0..4 (carry-maximizing pattern, mod p)
	allones := new(big.Int)
	for k := 4; k >= 0; k-- {
		allones.Lsh(allones, 52)
		allones.Or(allones, big.NewInt(mask52))
	}
	limb4 := new(big.Int).Lsh(one, 208) // lowest bit of the top limb
	pow255 := new(big.Int).Lsh(one, 255)
	maxu := new(big.Int).Sub(new(big.Int).Lsh(one, 256), one) // 2^256-1

	return [][32]byte{
		bi(big.NewInt(0)), bi(one), bi(big.NewInt(2)), bi(big.NewInt(977)),
		bi(pm1), bi(pm2), bi(half), bi(allones), bi(limb4), bi(pow255), bi(maxu),
	}
}

func TestEdgeCases(t *testing.T) {
	vals := edgeValues()
	for i, a := range vals {
		checkSqr(t, a)
		checkNegate(t, a)
		for j, b := range vals {
			checkMul(t, a, b)
			checkAdd(t, a, b)
			_ = i
			_ = j
		}
	}
}

// ---- algebraic properties (independent of btcec) ---------------------------
//
// These catch reduction bugs that uniform-random differential testing can miss,
// because they constrain the *structure* of the result rather than a value.

func TestProperties(t *testing.T) {
	mul := func(a, b [32]byte) [32]byte {
		var z Fe
		z.Mul(newFe(&a), newFe(&b))
		return z.Bytes()
	}
	add := func(a, b [32]byte) [32]byte {
		var z Fe
		z.Add(newFe(&a), newFe(&b))
		return z.Bytes()
	}
	one := bytesOf(big.NewInt(1))
	zero := bytesOf(big.NewInt(0))

	for i := 0; i < 5000; i++ {
		a, b, c := randBytes(t), randBytes(t), randBytes(t)

		if mul(a, b) != mul(b, a) {
			t.Fatalf("commutativity failed a=%x b=%x", a, b)
		}
		if got := mul(a, one); got != bytesOf(new(big.Int).Mod(new(big.Int).SetBytes(a[:]), P)) {
			t.Fatalf("identity a*1 != a, a=%x got=%x", a, got)
		}
		if got := mul(a, zero); got != zero {
			t.Fatalf("a*0 != 0, a=%x got=%x", a, got)
		}
		// distributivity: a*(b+c) == a*b + a*c
		lhs := mul(a, add(b, c))
		rhs := add(mul(a, b), mul(a, c))
		if lhs != rhs {
			t.Fatalf("distributivity failed\n a=%x\n b=%x\n c=%x", a, b, c)
		}
		// Sqr consistency: Sqr(a) == a*a
		var s Fe
		s.Sqr(newFe(&a))
		if s.Bytes() != mul(a, a) {
			t.Fatalf("Sqr(a) != a*a, a=%x", a)
		}
	}
}

// ---- differential fuzzing --------------------------------------------------

func FuzzMul(f *testing.F) {
	f.Add(make([]byte, 32), make([]byte, 32))
	f.Add([]byte{1}, []byte{2})
	f.Fuzz(func(t *testing.T, ab, bb []byte) {
		checkMul(t, coerce(ab), coerce(bb))
	})
}

func FuzzSqr(f *testing.F) {
	f.Add(make([]byte, 32))
	f.Add([]byte{2})
	f.Fuzz(func(t *testing.T, ab []byte) {
		checkSqr(t, coerce(ab))
	})
}
