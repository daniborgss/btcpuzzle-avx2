#ifndef RIPEMD160_SSE4_H
#define RIPEMD160_SSE4_H

#include <stdint.h>

// ripemd160_sse4_4 hashes 4 independent 32-byte messages in parallel using
// SSE2/SSE4 (one message per 32-bit SIMD lane). in points to 4*32 contiguous
// bytes (message l at in[l*32]); out receives 4*20 contiguous bytes (digest l
// at out[l*20]). Inputs are fixed 32-byte messages, so each is a single padded
// RIPEMD160 block.
void ripemd160_sse4_4(const uint8_t *in, uint8_t *out);

#endif
