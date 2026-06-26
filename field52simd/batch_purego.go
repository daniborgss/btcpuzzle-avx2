//go:build !avx512ifma

package field52simd

// Pure-Go fallback for the batched field ops: unpack the SoA lanes, run the
// single-lane scalar reference on each, repack. No cgo, builds anywhere. The
// IFMA backend (batch_ifma.go) replaces these with a single vector kernel call.

// MulBatch sets out[l] = a[l]*b[l] mod p for all Lanes lanes.
func MulBatch(out, a, b *Fe8) {
	var av, bv, ov [Lanes]Fe
	UnpackLanes(&av, a)
	UnpackLanes(&bv, b)
	for l := 0; l < Lanes; l++ {
		ov[l].Mul(&av[l], &bv[l])
	}
	PackLanes(out, &ov)
}

// SqrBatch sets out[l] = a[l]^2 mod p for all Lanes lanes.
func SqrBatch(out, a *Fe8) {
	var av, ov [Lanes]Fe
	UnpackLanes(&av, a)
	for l := 0; l < Lanes; l++ {
		ov[l].Sqr(&av[l])
	}
	PackLanes(out, &ov)
}

// Pure-Go fallback for the fused EC affine-add steps. The IFMA backend does
// each as a single cgo call over all groups; here they are plain Go loops over
// the existing batch ops (this is exactly what advance() used to inline).

// sqrN sets out = in^(2^n) for n >= 1 (out may alias in).
func sqrN(out, in *Fe8, n int) {
	SqrBatch(out, in)
	for i := 1; i < n; i++ {
		SqrBatch(out, out)
	}
}

// InverseFe8 sets out[l] = a[l]^(-1) mod p via Fermat (a^(p-2)) using the
// standard secp256k1 addition chain, all 8 lanes in parallel. a[l] == 0 -> 0.
func InverseFe8(out, a *Fe8) {
	var x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t Fe8
	SqrBatch(&x2, a)
	MulBatch(&x2, &x2, a)
	SqrBatch(&x3, &x2)
	MulBatch(&x3, &x3, a)
	sqrN(&x6, &x3, 3)
	MulBatch(&x6, &x6, &x3)
	sqrN(&x9, &x6, 3)
	MulBatch(&x9, &x9, &x3)
	sqrN(&x11, &x9, 2)
	MulBatch(&x11, &x11, &x2)
	sqrN(&x22, &x11, 11)
	MulBatch(&x22, &x22, &x11)
	sqrN(&x44, &x22, 22)
	MulBatch(&x44, &x44, &x22)
	sqrN(&x88, &x44, 44)
	MulBatch(&x88, &x88, &x44)
	sqrN(&x176, &x88, 88)
	MulBatch(&x176, &x176, &x88)
	sqrN(&x220, &x176, 44)
	MulBatch(&x220, &x220, &x44)
	sqrN(&x223, &x220, 3)
	MulBatch(&x223, &x223, &x3)
	sqrN(&t, &x223, 23)
	MulBatch(&t, &t, &x22)
	sqrN(&t, &t, 5)
	MulBatch(&t, &t, a)
	sqrN(&t, &t, 3)
	MulBatch(&t, &t, &x2)
	sqrN(&t, &t, 2)
	MulBatch(out, &t, a)
}

func oneFe8() Fe8 {
	var o Fe8
	for l := 0; l < Lanes; l++ {
		o[0][l] = 1
	}
	return o
}

// SlopeSetup: denom[g] = x[g] - xG, num[g] = y[g] - yG.
func SlopeSetup(denom, num, x, y []Fe8, xG, yG *Fe8) {
	for g := range x {
		SubBatch(&denom[g], &x[g], xG)
		SubBatch(&num[g], &y[g], yG)
	}
}

// MontForward: prefix[g] = product of denom[0..g-1]; accOut = full product.
func MontForward(prefix []Fe8, accOut *Fe8, denom []Fe8) {
	acc := oneFe8()
	for g := range denom {
		prefix[g] = acc
		MulBatch(&acc, &acc, &denom[g])
	}
	*accOut = acc
}

// MontBackward: inv[g] = invAcc*prefix[g]; invAcc *= denom[g] (reverse scan).
func MontBackward(inv []Fe8, invAcc *Fe8, prefix, denom []Fe8) {
	acc := *invAcc
	for g := len(denom) - 1; g >= 0; g-- {
		MulBatch(&inv[g], &acc, &prefix[g])
		MulBatch(&acc, &acc, &denom[g])
	}
	*invAcc = acc
}

// CanonBytes canonicalizes every lane of every group in src and writes its
// 32-byte big-endian value into dst (len(src)*Lanes*32 bytes).
func CanonBytes(dst []byte, src []Fe8) {
	for g := range src {
		var fe [Lanes]Fe
		UnpackLanes(&fe, &src[g])
		for l := 0; l < Lanes; l++ {
			b := fe[l].Bytes()
			copy(dst[(g*Lanes+l)*32:], b[:])
		}
	}
}

// PointAdd: second-pass affine add per group — lambda = num*inv,
// x3 = lambda^2 - x - xsub, y3 = lambda*(x - x3) - y.
func PointAdd(x, y, num, inv, xsub []Fe8) {
	for g := range x {
		var lambda, sq, x3, t, y3 Fe8
		MulBatch(&lambda, &num[g], &inv[g])
		SqrBatch(&sq, &lambda)
		SubBatch(&x3, &sq, &x[g])
		SubBatch(&x3, &x3, &xsub[g])
		SubBatch(&t, &x[g], &x3)
		MulBatch(&y3, &lambda, &t)
		SubBatch(&y3, &y3, &y[g])
		x[g] = x3
		y[g] = y3
	}
}
