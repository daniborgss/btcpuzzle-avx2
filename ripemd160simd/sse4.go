//go:build sse4 && !avx2

package ripemd160simd

/*
#cgo CFLAGS: -O3 -msse4.1 -DBACKEND_SSE4
#include "ripemd160_sse4.h"
*/
import "C"

import "unsafe"

// Lanes is how many messages HashBatch processes per call on the SSE backend.
// A 128-bit register holds four 32-bit lanes, hence 4-way.
const Lanes = 4

// HashBatch computes RIPEMD160 of Lanes 32-byte messages in parallel using the
// 4-way SSE2/SSE4 implementation. out[l] is the digest of in[l].
func HashBatch(out *[Lanes][20]byte, in *[Lanes][32]byte) {
	C.ripemd160_sse4_4(
		(*C.uint8_t)(unsafe.Pointer(&in[0][0])),
		(*C.uint8_t)(unsafe.Pointer(&out[0][0])),
	)
}
