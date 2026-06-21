#ifndef RIPEMD160_AVX2_H
#define RIPEMD160_AVX2_H

#include <stdint.h>

// ripemd160_avx2_8 hashes 8 independent 32-byte messages in parallel using
// AVX2 (one message per 32-bit SIMD lane). in points to 8*32 contiguous bytes
// (message l at in[l*32]); out receives 8*20 contiguous bytes (digest l at
// out[l*20]). Inputs are fixed 32-byte messages, so each is a single padded
// RIPEMD160 block.
void ripemd160_avx2_8(const uint8_t *in, uint8_t *out);

#endif
