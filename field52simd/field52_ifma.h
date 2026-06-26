#ifndef FIELD52_IFMA_H
#define FIELD52_IFMA_H

#include <stdint.h>

// 8-way secp256k1 field ops over the SoA layout Fe8 ([5][8]uint64): 5 limbs,
// each a contiguous block of 8 lanes (one __m512i). Pointers reference 40
// contiguous uint64 (limb k's 8 lanes at base + 8*k). Inputs limbs < 2^52
// (limb 4 < 2^48); outputs are magnitude-1 but may be denormalized.

// field52_mul8: out[l] = a[l] * b[l] mod p, for the 8 lanes.
void field52_mul8(uint64_t *out, const uint64_t *a, const uint64_t *b);

// field52_sqr8: out[l] = a[l]^2 mod p, for the 8 lanes (specialized squaring).
void field52_sqr8(uint64_t *out, const uint64_t *a);

// field52_inverse8: out[l] = a[l]^(-1) mod p (Fermat chain, one cgo call).
void field52_inverse8(uint64_t *out, const uint64_t *a);

// field52_canon_bytes8: canonicalize ng groups and pack each lane to 32
// big-endian bytes (out is ng*8*32 bytes).
void field52_canon_bytes8(uint8_t *out, const uint64_t *in, long ng);

// Fused per-pass steps over ng groups (one cgo call for the whole laneSet).
void field52_slope_setup8(uint64_t *denom, uint64_t *num, const uint64_t *x,
                          const uint64_t *y, const uint64_t *xG,
                          const uint64_t *yG, long ng);
void field52_mont_forward8(uint64_t *prefix, uint64_t *accOut,
                           const uint64_t *denom, long ng);
void field52_mont_backward8(uint64_t *inv, uint64_t *invAcc,
                            const uint64_t *prefix, const uint64_t *denom,
                            long ng);
void field52_point_add8(uint64_t *x, uint64_t *y, const uint64_t *num,
                        const uint64_t *inv, const uint64_t *xsub, long ng);

#endif
