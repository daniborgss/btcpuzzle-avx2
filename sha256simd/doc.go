// Package sha256simd computes SHA-256 of a batch of 8 fixed 33-byte messages
// (a compressed secp256k1 public key: 0x02/0x03 || 32-byte X), with a
// build-tag-selected backend mirroring ripemd160simd:
//
//   - shani.go + sha256_shani.c (-tags shani, cgo): the SHA-NI extension, one
//     hash at a time, single padded block; one cgo call for all 8 lanes.
//   - purego.go (no tag): crypto/sha256.Sum256 per lane.
//
// Both expose `const Lanes` and `HashBatch(out *[Lanes][32]byte, in *[Lanes][33]byte)`.
// The SHA-NI path drops the stdlib hash.Hash framework overhead (Reset/Write/Sum
// per lane) that dominated the search loop once the field math was vectorized;
// sha256simd_test.go checks the backend byte-for-byte against crypto/sha256.
package sha256simd
