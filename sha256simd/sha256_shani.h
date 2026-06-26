#ifndef SHA256_SHANI_H
#define SHA256_SHANI_H

#include <stdint.h>

// sha256_pubkey8 computes SHA-256 of 8 fixed 33-byte messages using the SHA-NI
// extension (one hash at a time; single padded block each). in points to 8*33
// contiguous bytes (message l at in[l*33]); out receives 8*32 bytes (digest l at
// out[l*32]).
void sha256_pubkey8(uint8_t *out, const uint8_t *in);

#endif
