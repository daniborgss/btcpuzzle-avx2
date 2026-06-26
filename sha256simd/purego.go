//go:build !shani

package sha256simd

import "crypto/sha256"

// Lanes is how many messages HashBatch processes per call.
const Lanes = 8

// HashBatch computes SHA-256 of Lanes 33-byte messages, one per lane, using the
// standard library (no SHA-NI kernel). out[l] is the digest of in[l].
func HashBatch(out *[Lanes][32]byte, in *[Lanes][33]byte) {
	for l := 0; l < Lanes; l++ {
		out[l] = sha256.Sum256(in[l][:])
	}
}
