// Package ripemd160simd is an isolated prototype: an 8-way AVX2 multi-message
// RIPEMD160 for fixed 32-byte inputs, used to measure the achievable speedup
// over the pure-Go single-message hasher before wiring it into the search loop.
package ripemd160simd

/*
#cgo CFLAGS: -O3 -mavx2
#include "ripemd160_avx2.h"
*/
import "C"

import "unsafe"

// Lanes is how many messages Hash8 processes per call.
const Lanes = 8

// Hash8 computes RIPEMD160 of eight 32-byte messages in parallel. out[l] is the
// digest of in[l].
func Hash8(out *[Lanes][20]byte, in *[Lanes][32]byte) {
	C.ripemd160_avx2_8(
		(*C.uint8_t)(unsafe.Pointer(&in[0][0])),
		(*C.uint8_t)(unsafe.Pointer(&out[0][0])),
	)
}
