//go:build avx512ifma

package field52simd

/*
#cgo CFLAGS: -O3 -mavx512f -mavx512vl -mavx512ifma -DBACKEND_AVX512IFMA
#include "field52_ifma.h"
*/
import "C"

import "unsafe"

// MulBatch sets out[l] = a[l]*b[l] mod p for all Lanes lanes, using the 8-way
// AVX-512 IFMA kernel. Fe8 is memory-identical to the C [5][8]uint64 the kernel
// loads as 5 __m512i.
func MulBatch(out, a, b *Fe8) {
	C.field52_mul8(
		(*C.uint64_t)(unsafe.Pointer(&out[0][0])),
		(*C.uint64_t)(unsafe.Pointer(&a[0][0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0][0])),
	)
}

// SqrBatch sets out[l] = a[l]^2 mod p for all Lanes lanes (specialized squaring).
func SqrBatch(out, a *Fe8) {
	C.field52_sqr8(
		(*C.uint64_t)(unsafe.Pointer(&out[0][0])),
		(*C.uint64_t)(unsafe.Pointer(&a[0][0])),
	)
}

// InverseFe8 runs the whole secp256k1 Fermat inverse chain in one cgo call.
func InverseFe8(out, a *Fe8) {
	C.field52_inverse8(
		(*C.uint64_t)(unsafe.Pointer(&out[0][0])),
		(*C.uint64_t)(unsafe.Pointer(&a[0][0])),
	)
}

// CanonBytes canonicalizes every lane of every group in src and writes its
// 32-byte big-endian value into dst (len(src)*Lanes*32 bytes), in one cgo call.
func CanonBytes(dst []byte, src []Fe8) {
	C.field52_canon_bytes8(
		(*C.uint8_t)(unsafe.Pointer(&dst[0])),
		(*C.uint64_t)(unsafe.Pointer(&src[0])),
		C.long(len(src)),
	)
}

// The fused EC affine-add steps below process a whole laneSet (one cgo call for
// all groups) — see the matching doc in batch_purego.go. []Fe8 is contiguous,
// so &s[0] is the base of len(s)*40 uint64.

func SlopeSetup(denom, num, x, y []Fe8, xG, yG *Fe8) {
	C.field52_slope_setup8(
		(*C.uint64_t)(unsafe.Pointer(&denom[0])),
		(*C.uint64_t)(unsafe.Pointer(&num[0])),
		(*C.uint64_t)(unsafe.Pointer(&x[0])),
		(*C.uint64_t)(unsafe.Pointer(&y[0])),
		(*C.uint64_t)(unsafe.Pointer(xG)),
		(*C.uint64_t)(unsafe.Pointer(yG)),
		C.long(len(x)),
	)
}

func MontForward(prefix []Fe8, accOut *Fe8, denom []Fe8) {
	C.field52_mont_forward8(
		(*C.uint64_t)(unsafe.Pointer(&prefix[0])),
		(*C.uint64_t)(unsafe.Pointer(accOut)),
		(*C.uint64_t)(unsafe.Pointer(&denom[0])),
		C.long(len(denom)),
	)
}

func MontBackward(inv []Fe8, invAcc *Fe8, prefix, denom []Fe8) {
	C.field52_mont_backward8(
		(*C.uint64_t)(unsafe.Pointer(&inv[0])),
		(*C.uint64_t)(unsafe.Pointer(invAcc)),
		(*C.uint64_t)(unsafe.Pointer(&prefix[0])),
		(*C.uint64_t)(unsafe.Pointer(&denom[0])),
		C.long(len(denom)),
	)
}

func PointAdd(x, y, num, inv, xsub []Fe8) {
	C.field52_point_add8(
		(*C.uint64_t)(unsafe.Pointer(&x[0])),
		(*C.uint64_t)(unsafe.Pointer(&y[0])),
		(*C.uint64_t)(unsafe.Pointer(&num[0])),
		(*C.uint64_t)(unsafe.Pointer(&inv[0])),
		(*C.uint64_t)(unsafe.Pointer(&xsub[0])),
		C.long(len(x)),
	)
}
