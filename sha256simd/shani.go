//go:build shani

package sha256simd

/*
// -msha is not on cgo's CFLAGS allowlist, so the SHA-NI feature is enabled per
// function via __attribute__((target(...))) in the .c instead of a -m flag.
#cgo CFLAGS: -O3 -DBACKEND_SHANI
#include "sha256_shani.h"
*/
import "C"

import "unsafe"

// Lanes is how many messages HashBatch processes per call on the SHA-NI backend.
const Lanes = 8

// HashBatch computes SHA-256 of Lanes 33-byte messages using the SHA-NI
// extension in a single cgo call. out[l] is the digest of in[l].
func HashBatch(out *[Lanes][32]byte, in *[Lanes][33]byte) {
	C.sha256_pubkey8(
		(*C.uint8_t)(unsafe.Pointer(&out[0][0])),
		(*C.uint8_t)(unsafe.Pointer(&in[0][0])),
	)
}
