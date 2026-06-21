#include "ripemd160_avx2.h"
#include <immintrin.h>
#include <string.h>

// 8-way (AVX2) multi-message RIPEMD160 for fixed 32-byte inputs.
//
// Each 32-bit SIMD lane carries one independent message, so the algorithm is
// the scalar RIPEMD160 with every 32-bit word op replaced by its __m256i
// equivalent. Control flow (round functions, message-word order, rotation
// amounts) is identical across lanes, which is exactly what makes multi-buffer
// SIMD hashing efficient.

#define ADD(a, b) _mm256_add_epi32((a), (b))
#define XOR(a, b) _mm256_xor_si256((a), (b))
#define AND(a, b) _mm256_and_si256((a), (b))
#define OR(a, b) _mm256_or_si256((a), (b))
#define ANDNOT(a, b) _mm256_andnot_si256((a), (b)) // (~a) & b
#define NOTV(x) _mm256_xor_si256((x), ones)

// Round boolean functions f1..f5 (RIPEMD160).
#define F1(b, c, d) XOR(XOR((b), (c)), (d))
#define F2(b, c, d) OR(AND((b), (c)), ANDNOT((b), (d)))
#define F3(b, c, d) XOR(OR((b), NOTV(c)), (d))
#define F4(b, c, d) OR(AND((b), (d)), ANDNOT((d), (c)))
#define F5(b, c, d) XOR((b), OR((c), NOTV(d)))

// Rotate-left each 32-bit lane by a runtime amount n (same n for all lanes).
static inline __m256i rolv(__m256i x, int n) {
    __m128i l = _mm_cvtsi32_si128(n);
    __m128i r = _mm_cvtsi32_si128(32 - n);
    return _mm256_or_si256(_mm256_sll_epi32(x, l), _mm256_srl_epi32(x, r));
}

// One RIPEMD160 step with in-place role rotation of (a,b,c,d,e). F is one of
// the f-macros; x is the selected message word, k the round constant, s the
// rotation amount. After the step the variables are permuted so the next call
// uses the canonical (e,a,b,c,d) ordering.
#define STEP(F, a, b, c, d, e, x, k, s)            \
    do {                                           \
        __m256i t = ADD((a), F((b), (c), (d)));    \
        t = ADD(t, (x));                           \
        t = ADD(t, (k));                           \
        t = ADD(rolv(t, (s)), (e));                \
        (c) = rolv((c), 10);                       \
        (a) = (e); (e) = (d); (d) = (c); (c) = (b); (b) = t; \
    } while (0)

// Message-word order and rotation tables (left line: rL/sL, right line: rR/sR).
static const int rL[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13};
static const int rR[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11};
static const int sL[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6};
static const int sR[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11};

void ripemd160_avx2_8(const uint8_t *in, uint8_t *out) {
    const __m256i ones = _mm256_set1_epi32(-1);

    // Round constants broadcast across all lanes.
    const __m256i KL0 = _mm256_set1_epi32(0x00000000);
    const __m256i KL1 = _mm256_set1_epi32(0x5A827999);
    const __m256i KL2 = _mm256_set1_epi32(0x6ED9EBA1);
    const __m256i KL3 = _mm256_set1_epi32(0x8F1BBCDC);
    const __m256i KL4 = _mm256_set1_epi32(0xA953FD4E);
    const __m256i KR0 = _mm256_set1_epi32(0x50A28BE6);
    const __m256i KR1 = _mm256_set1_epi32(0x5C4DD124);
    const __m256i KR2 = _mm256_set1_epi32(0x6D703EF3);
    const __m256i KR3 = _mm256_set1_epi32(0x7A6D76E9);
    const __m256i KR4 = _mm256_set1_epi32(0x00000000);

    // Transpose the 8 messages into 16 message-word vectors. Words 0..7 hold
    // the per-lane data; words 8..15 are the fixed single-block padding for a
    // 32-byte (256-bit) message.
    __m256i X[16];
    for (int j = 0; j < 8; j++) {
        uint32_t w[8];
        for (int l = 0; l < 8; l++) {
            memcpy(&w[l], in + l * 32 + j * 4, 4); // little-endian load
        }
        X[j] = _mm256_setr_epi32(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
    }
    X[8] = _mm256_set1_epi32(0x00000080);  // 0x80 padding byte
    X[9] = _mm256_setzero_si256();
    X[10] = _mm256_setzero_si256();
    X[11] = _mm256_setzero_si256();
    X[12] = _mm256_setzero_si256();
    X[13] = _mm256_setzero_si256();
    X[14] = _mm256_set1_epi32(0x00000100); // bit length = 256
    X[15] = _mm256_setzero_si256();

    const __m256i h0 = _mm256_set1_epi32(0x67452301);
    const __m256i h1 = _mm256_set1_epi32(0xEFCDAB89);
    const __m256i h2 = _mm256_set1_epi32(0x98BADCFE);
    const __m256i h3 = _mm256_set1_epi32(0x10325476);
    const __m256i h4 = _mm256_set1_epi32(0xC3D2E1F0);

    // Left line.
    __m256i al = h0, bl = h1, cl = h2, dl = h3, el = h4;
    for (int i = 0; i < 16; i++) STEP(F1, al, bl, cl, dl, el, X[rL[i]], KL0, sL[i]);
    for (int i = 16; i < 32; i++) STEP(F2, al, bl, cl, dl, el, X[rL[i]], KL1, sL[i]);
    for (int i = 32; i < 48; i++) STEP(F3, al, bl, cl, dl, el, X[rL[i]], KL2, sL[i]);
    for (int i = 48; i < 64; i++) STEP(F4, al, bl, cl, dl, el, X[rL[i]], KL3, sL[i]);
    for (int i = 64; i < 80; i++) STEP(F5, al, bl, cl, dl, el, X[rL[i]], KL4, sL[i]);

    // Right line.
    __m256i ar = h0, br = h1, cr = h2, dr = h3, er = h4;
    for (int i = 0; i < 16; i++) STEP(F5, ar, br, cr, dr, er, X[rR[i]], KR0, sR[i]);
    for (int i = 16; i < 32; i++) STEP(F4, ar, br, cr, dr, er, X[rR[i]], KR1, sR[i]);
    for (int i = 32; i < 48; i++) STEP(F3, ar, br, cr, dr, er, X[rR[i]], KR2, sR[i]);
    for (int i = 48; i < 64; i++) STEP(F2, ar, br, cr, dr, er, X[rR[i]], KR3, sR[i]);
    for (int i = 64; i < 80; i++) STEP(F1, ar, br, cr, dr, er, X[rR[i]], KR4, sR[i]);

    // Combine the two lines with the (single-block) chaining values.
    __m256i t = ADD(ADD(h1, cl), dr);
    __m256i n1 = ADD(ADD(h2, dl), er);
    __m256i n2 = ADD(ADD(h3, el), ar);
    __m256i n3 = ADD(ADD(h4, al), br);
    __m256i n4 = ADD(ADD(h0, bl), cr);
    __m256i n0 = t;

    // Scatter digests back to per-lane little-endian byte order.
    uint32_t o0[8], o1[8], o2[8], o3[8], o4[8];
    _mm256_storeu_si256((__m256i *)o0, n0);
    _mm256_storeu_si256((__m256i *)o1, n1);
    _mm256_storeu_si256((__m256i *)o2, n2);
    _mm256_storeu_si256((__m256i *)o3, n3);
    _mm256_storeu_si256((__m256i *)o4, n4);
    for (int l = 0; l < 8; l++) {
        uint8_t *p = out + l * 20;
        memcpy(p + 0, &o0[l], 4);
        memcpy(p + 4, &o1[l], 4);
        memcpy(p + 8, &o2[l], 4);
        memcpy(p + 12, &o3[l], 4);
        memcpy(p + 16, &o4[l], 4);
    }
}
