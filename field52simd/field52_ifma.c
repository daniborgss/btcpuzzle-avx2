//go:build avx512ifma

// Compiled only for the AVX-512 IFMA backend. Unlike ripemd160simd (whose
// package always has an active cgo file under any cgo build), field52simd has
// no cgo file unless the avx512ifma tag is set, so this .c would be an orphan
// under e.g. -tags avx2 alone. The build constraint above excludes it in that
// case; the #ifdef is a second guard (cgo otherwise compiles every .c).
// BACKEND_AVX512IFMA is defined by batch_ifma.go's cgo CFLAGS.
#ifdef BACKEND_AVX512IFMA

#include "field52_ifma.h"
#include <immintrin.h>

// 8-way secp256k1 field arithmetic in radix 2^52, one element per 64-bit lane.
// This is a literal, op-for-op translation of the scalar reference in
// purego.go (madd52lo/hi, the schoolbook column schedule, and the 3-round
// Solinas reduction) with every op replaced by its __m512i / vpmadd52
// equivalent. The Go reference is the byte-for-byte oracle (see batch_test.go).

#define VLO(acc, a, b) _mm512_madd52lo_epu64((acc), (a), (b)) // acc += low52(a*b)
#define VHI(acc, a, b) _mm512_madd52hi_epu64((acc), (a), (b)) // acc += hi52 (a*b)
#define FE8 40 // uint64 per Fe8 group (5 limbs * 8 lanes)

// reduce folds the 10 product columns t[0..9] (each lane < 2^52 after the carry
// pass) down to 5 limbs r[0..4] congruent mod p. Mirrors reduceSolinas:
//   c   = 2^32 + 977         (fold at the 2^256 boundary)
//   16c = 68719492368        (fold the high limbs, weight starts at 2^260)
static void reduce(__m512i r[5], const __m512i t[10]) {
    const __m512i mask52 = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i mask48 = _mm512_set1_epi64((1ULL << 48) - 1);
    const __m512i cAdj = _mm512_set1_epi64(68719492368ULL);
    const __m512i c52 = _mm512_set1_epi64(4294968273ULL);
    __m512i m;

    // ---- Round 1: fold columns 5..9 (weight >= 2^260) with 16c. ----
    m = VLO(t[0], t[5], cAdj);
    r[0] = _mm512_and_si512(m, mask52);

    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), t[1]);
    m = VHI(m, t[5], cAdj);
    m = VLO(m, t[6], cAdj);
    r[1] = _mm512_and_si512(m, mask52);

    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), t[2]);
    m = VHI(m, t[6], cAdj);
    m = VLO(m, t[7], cAdj);
    r[2] = _mm512_and_si512(m, mask52);

    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), t[3]);
    m = VHI(m, t[7], cAdj);
    m = VLO(m, t[8], cAdj);
    r[3] = _mm512_and_si512(m, mask52);

    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), t[4]);
    m = VHI(m, t[8], cAdj);
    m = VLO(m, t[9], cAdj);
    r[4] = _mm512_and_si512(m, mask52);

    __m512i r5 = _mm512_srli_epi64(m, 52); // column 5 (weight 2^260), small
    r5 = VHI(r5, t[9], cAdj);

    // ---- Round 2: fold r5 (weight 2^260) with 16c. ----
    m = VLO(r[0], r5, cAdj);
    r[0] = _mm512_and_si512(m, mask52);
    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), r[1]);
    m = VHI(m, r5, cAdj);
    r[1] = _mm512_and_si512(m, mask52);
    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), r[2]);
    r[2] = _mm512_and_si512(m, mask52);
    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), r[3]);
    r[3] = _mm512_and_si512(m, mask52);
    r[4] = _mm512_add_epi64(_mm512_srli_epi64(m, 52), r[4]);

    // ---- Round 3: fold the final overflow above 2^256 with c (unscaled). ----
    __m512i mtop = _mm512_srli_epi64(r[4], 48);
    r[4] = _mm512_and_si512(r[4], mask48);
    m = VLO(r[0], mtop, c52);
    r[0] = _mm512_and_si512(m, mask52);
    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), r[1]);
    m = VHI(m, mtop, c52);
    r[1] = _mm512_and_si512(m, mask52);
    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), r[2]);
    r[2] = _mm512_and_si512(m, mask52);
    m = _mm512_add_epi64(_mm512_srli_epi64(m, 52), r[3]);
    r[3] = _mm512_and_si512(m, mask52);
    r[4] = _mm512_add_epi64(_mm512_srli_epi64(m, 52), r[4]);
}

// carry_normalize reduces the 10 schoolbook columns to radix 2^52 (each < 2^52).
static void carry_normalize(__m512i t[10]) {
    const __m512i mask52 = _mm512_set1_epi64((1ULL << 52) - 1);
    __m512i carry = _mm512_setzero_si512();
    for (int k = 0; k < 10; k++) {
        __m512i v = _mm512_add_epi64(t[k], carry);
        t[k] = _mm512_and_si512(v, mask52);
        carry = _mm512_srli_epi64(v, 52);
    }
}

static inline void load5(__m512i r[5], const uint64_t *p) {
    for (int k = 0; k < 5; k++) {
        r[k] = _mm512_loadu_si512((const void *)(p + 8 * k));
    }
}

static inline void store5(uint64_t *p, const __m512i r[5]) {
    for (int k = 0; k < 5; k++) {
        _mm512_storeu_si512((void *)(p + 8 * k), r[k]);
    }
}

// mul_lanes/sqr_lanes/sub_lanes operate register-to-register so the fused EC
// steps below chain them with no intermediate loads/stores (and no cgo call).
static void mul_lanes(__m512i r[5], const __m512i av[5], const __m512i bv[5]) {
    __m512i t[10];
    for (int k = 0; k < 10; k++) {
        t[k] = _mm512_setzero_si512();
    }
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            t[i + j] = VLO(t[i + j], av[i], bv[j]);
            t[i + j + 1] = VHI(t[i + j + 1], av[i], bv[j]);
        }
    }
    carry_normalize(t);
    reduce(r, t);
}

static void sqr_lanes(__m512i r[5], const __m512i av[5]) {
    __m512i t[10];
    for (int k = 0; k < 10; k++) {
        t[k] = _mm512_setzero_si512();
    }
    for (int i = 0; i < 5; i++) {
        for (int j = i + 1; j < 5; j++) {
            t[i + j] = VLO(t[i + j], av[i], av[j]);
            t[i + j + 1] = VHI(t[i + j + 1], av[i], av[j]);
        }
    }
    for (int k = 0; k < 10; k++) {
        t[k] = _mm512_slli_epi64(t[k], 1);
    }
    for (int i = 0; i < 5; i++) {
        t[2 * i] = VLO(t[2 * i], av[i], av[i]);
        t[2 * i + 1] = VHI(t[2 * i + 1], av[i], av[i]);
    }
    carry_normalize(t);
    reduce(r, t);
}

// sub_lanes: r = a - b mod p (denormalized), via a + 4p - b then a top fold.
static void sub_lanes(__m512i r[5], const __m512i av[5], const __m512i bv[5]) {
    const __m512i mask52 = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i mask48 = _mm512_set1_epi64((1ULL << 48) - 1);
    const __m512i c52 = _mm512_set1_epi64(4294968273ULL);
    const __m512i nb[5] = {
        _mm512_set1_epi64(4ULL * 0xFFFFEFFFFFC2FULL),
        _mm512_set1_epi64(4ULL * 0xFFFFFFFFFFFFFULL),
        _mm512_set1_epi64(4ULL * 0xFFFFFFFFFFFFFULL),
        _mm512_set1_epi64(4ULL * 0xFFFFFFFFFFFFFULL),
        _mm512_set1_epi64(4ULL * 0x0FFFFFFFFFFFFULL),
    };
    __m512i carry = _mm512_setzero_si512();
    for (int k = 0; k < 5; k++) {
        __m512i v = _mm512_add_epi64(_mm512_add_epi64(av[k], nb[k]), carry);
        v = _mm512_sub_epi64(v, bv[k]);
        r[k] = _mm512_and_si512(v, mask52);
        carry = _mm512_srli_epi64(v, 52);
    }
    // Fold the overflow above 2^256 (limb 4 >> 48) via c.
    __m512i mtop = _mm512_srli_epi64(r[4], 48);
    r[4] = _mm512_and_si512(r[4], mask48);
    __m512i v = VLO(r[0], mtop, c52);
    r[0] = _mm512_and_si512(v, mask52);
    __m512i cc = _mm512_srli_epi64(v, 52);
    for (int k = 1; k < 5; k++) {
        v = _mm512_add_epi64(r[k], cc);
        r[k] = _mm512_and_si512(v, mask52);
        cc = _mm512_srli_epi64(v, 52);
    }
}

// canon_lanes reduces r in place to the unique canonical form (value < p), the
// vector analog of purego.go's reduceCanonical. Inputs are PointAdd/sub outputs
// (limbs < 2^52, value < 2^256 + tiny), so two top folds then one conditional
// subtract of p suffice.
static void canon_lanes(__m512i r[5]) {
    const __m512i mask52 = _mm512_set1_epi64((1ULL << 52) - 1);
    const __m512i mask48 = _mm512_set1_epi64((1ULL << 48) - 1);
    const __m512i c52 = _mm512_set1_epi64(4294968273ULL);
    const __m512i bit52 = _mm512_set1_epi64(1ULL << 52);
    const __m512i one = _mm512_set1_epi64(1);
    const __m512i pv[5] = {
        _mm512_set1_epi64(0xFFFFEFFFFFC2FULL), _mm512_set1_epi64(0xFFFFFFFFFFFFFULL),
        _mm512_set1_epi64(0xFFFFFFFFFFFFFULL), _mm512_set1_epi64(0xFFFFFFFFFFFFFULL),
        _mm512_set1_epi64(0x0FFFFFFFFFFFFULL),
    };
    __m512i carry = _mm512_setzero_si512();
    for (int k = 0; k < 5; k++) {
        __m512i v = _mm512_add_epi64(r[k], carry);
        r[k] = _mm512_and_si512(v, mask52);
        carry = _mm512_srli_epi64(v, 52);
    }
    for (int it = 0; it < 2; it++) {
        __m512i mtop = _mm512_srli_epi64(r[4], 48);
        r[4] = _mm512_and_si512(r[4], mask48);
        __m512i v = VLO(r[0], mtop, c52);
        r[0] = _mm512_and_si512(v, mask52);
        __m512i cc = _mm512_srli_epi64(v, 52);
        for (int k = 1; k < 5; k++) {
            v = _mm512_add_epi64(r[k], cc);
            r[k] = _mm512_and_si512(v, mask52);
            cc = _mm512_srli_epi64(v, 52);
        }
    }
    // Conditional subtract p: t = r - p; where it didn't borrow (r >= p), use t.
    __m512i borrow = _mm512_setzero_si512(), t[5];
    for (int k = 0; k < 5; k++) {
        __m512i d = _mm512_sub_epi64(
            _mm512_sub_epi64(_mm512_or_si512(r[k], bit52), pv[k]), borrow);
        t[k] = _mm512_and_si512(d, mask52);
        borrow = _mm512_sub_epi64(one, _mm512_and_si512(_mm512_srli_epi64(d, 52), one));
    }
    __mmask8 ge = _mm512_cmpeq_epi64_mask(borrow, _mm512_setzero_si512());
    for (int k = 0; k < 5; k++) {
        r[k] = _mm512_mask_blend_epi64(ge, r[k], t[k]);
    }
}

static inline void put_be64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        p[i] = (uint8_t)(v >> (56 - 8 * i));
    }
}

// field52_canon_bytes8: canonicalize ng groups and pack each lane's value to 32
// big-endian bytes (out is ng*8*32 bytes; lane g*8+l at offset (g*8+l)*32).
void field52_canon_bytes8(uint8_t *out, const uint64_t *in, long ng) {
    for (long g = 0; g < ng; g++) {
        __m512i r[5];
        load5(r, in + FE8 * g);
        canon_lanes(r);
        uint64_t lanes[5][8];
        for (int k = 0; k < 5; k++) {
            _mm512_storeu_si512((void *)lanes[k], r[k]);
        }
        for (int l = 0; l < 8; l++) {
            uint64_t l0 = lanes[0][l], l1 = lanes[1][l], l2 = lanes[2][l];
            uint64_t l3 = lanes[3][l], l4 = lanes[4][l];
            uint64_t w0 = l0 | (l1 << 52);
            uint64_t w1 = (l1 >> 12) | (l2 << 40);
            uint64_t w2 = (l2 >> 24) | (l3 << 28);
            uint64_t w3 = (l3 >> 36) | (l4 << 16);
            uint8_t *o = out + (g * 8 + l) * 32;
            put_be64(o + 0, w3);
            put_be64(o + 8, w2);
            put_be64(o + 16, w1);
            put_be64(o + 24, w0);
        }
    }
}

void field52_mul8(uint64_t *out, const uint64_t *a, const uint64_t *b) {
    __m512i av[5], bv[5], r[5];
    load5(av, a);
    load5(bv, b);
    mul_lanes(r, av, bv);
    store5(out, r);
}

void field52_sqr8(uint64_t *out, const uint64_t *a) {
    __m512i av[5], r[5];
    load5(av, a);
    sqr_lanes(r, av);
    store5(out, r);
}

// sqrn_lanes: r = a^(2^n), n >= 1 (r may alias a).
static void sqrn_lanes(__m512i r[5], const __m512i a[5], int n) {
    sqr_lanes(r, a);
    for (int i = 1; i < n; i++) {
        sqr_lanes(r, r);
    }
}

// inv_lanes: r = a^(p-2) mod p (modular inverse) via the standard secp256k1
// addition chain — the same chain as the Go InverseFe8, but the whole ~270-op
// sequence runs in one cgo call instead of one per Mul/Sqr.
static void inv_lanes(__m512i r[5], const __m512i a[5]) {
    __m512i x2[5], x3[5], x6[5], x9[5], x11[5], x22[5], x44[5];
    __m512i x88[5], x176[5], x220[5], x223[5], t[5];
    sqr_lanes(x2, a);       mul_lanes(x2, x2, a);
    sqr_lanes(x3, x2);      mul_lanes(x3, x3, a);
    sqrn_lanes(x6, x3, 3);  mul_lanes(x6, x6, x3);
    sqrn_lanes(x9, x6, 3);  mul_lanes(x9, x9, x3);
    sqrn_lanes(x11, x9, 2); mul_lanes(x11, x11, x2);
    sqrn_lanes(x22, x11, 11);  mul_lanes(x22, x22, x11);
    sqrn_lanes(x44, x22, 22);  mul_lanes(x44, x44, x22);
    sqrn_lanes(x88, x44, 44);  mul_lanes(x88, x88, x44);
    sqrn_lanes(x176, x88, 88); mul_lanes(x176, x176, x88);
    sqrn_lanes(x220, x176, 44); mul_lanes(x220, x220, x44);
    sqrn_lanes(x223, x220, 3);  mul_lanes(x223, x223, x3);
    sqrn_lanes(t, x223, 23); mul_lanes(t, t, x22);
    sqrn_lanes(t, t, 5);     mul_lanes(t, t, a);
    sqrn_lanes(t, t, 3);     mul_lanes(t, t, x2);
    sqrn_lanes(t, t, 2);     mul_lanes(r, t, a);
}

void field52_inverse8(uint64_t *out, const uint64_t *a) {
    __m512i av[5], r[5];
    load5(av, a);
    inv_lanes(r, av);
    store5(out, r);
}

// ---- Fused per-pass steps over a whole laneSet (ng groups of 8) -------------
//
// Each takes one cgo call for all ng groups instead of one per group per op,
// which is what actually matters at this point: cgo call overhead, not the SIMD
// itself, dominated the loop. A group occupies 40 contiguous uint64 (5 limbs *
// 8 lanes), so group g lives at offset 40*g. Mirrors advance() in search.go.

// slope_setup: denom = x - xG, num = y - yG, for every group.
void field52_slope_setup8(uint64_t *denom, uint64_t *num, const uint64_t *x,
                          const uint64_t *y, const uint64_t *xG,
                          const uint64_t *yG, long ng) {
    __m512i XG[5], YG[5];
    load5(XG, xG);
    load5(YG, yG);
    for (long g = 0; g < ng; g++) {
        __m512i X[5], Y[5], r[5];
        load5(X, x + FE8 * g);
        load5(Y, y + FE8 * g);
        sub_lanes(r, X, XG);
        store5(denom + FE8 * g, r);
        sub_lanes(r, Y, YG);
        store5(num + FE8 * g, r);
    }
}

// mont_forward: prefix[g] = product of denom[0..g-1]; accOut = full product.
void field52_mont_forward8(uint64_t *prefix, uint64_t *accOut,
                           const uint64_t *denom, long ng) {
    __m512i acc[5];
    acc[0] = _mm512_set1_epi64(1);
    for (int k = 1; k < 5; k++) {
        acc[k] = _mm512_setzero_si512();
    }
    for (long g = 0; g < ng; g++) {
        store5(prefix + FE8 * g, acc);
        __m512i D[5], r[5];
        load5(D, denom + FE8 * g);
        mul_lanes(r, acc, D);
        for (int k = 0; k < 5; k++) {
            acc[k] = r[k];
        }
    }
    store5(accOut, acc);
}

// mont_backward: inv[g] = invAcc * prefix[g]; invAcc *= denom[g] (reverse scan).
void field52_mont_backward8(uint64_t *inv, uint64_t *invAcc,
                            const uint64_t *prefix, const uint64_t *denom,
                            long ng) {
    __m512i IA[5];
    load5(IA, invAcc);
    for (long g = ng - 1; g >= 0; g--) {
        __m512i P[5], D[5], r[5];
        load5(P, prefix + FE8 * g);
        mul_lanes(r, IA, P);
        store5(inv + FE8 * g, r);
        load5(D, denom + FE8 * g);
        mul_lanes(r, IA, D);
        for (int k = 0; k < 5; k++) {
            IA[k] = r[k];
        }
    }
    store5(invAcc, IA);
}

// point_add: the second-pass affine add for every group, in registers:
//   lambda = num*inv; x3 = lambda^2 - x - xsub; y3 = lambda*(x - x3) - y.
void field52_point_add8(uint64_t *x, uint64_t *y, const uint64_t *num,
                        const uint64_t *inv, const uint64_t *xsub, long ng) {
    for (long g = 0; g < ng; g++) {
        __m512i X[5], Y[5], N[5], I[5], XS[5], L[5], S[5], X3[5], T[5], Y3[5];
        load5(X, x + FE8 * g);
        load5(Y, y + FE8 * g);
        load5(N, num + FE8 * g);
        load5(I, inv + FE8 * g);
        load5(XS, xsub + FE8 * g);
        mul_lanes(L, N, I);   // lambda
        sqr_lanes(S, L);      // lambda^2
        sub_lanes(X3, S, X);  // - x
        sub_lanes(X3, X3, XS); // - xsub
        sub_lanes(T, X, X3);  // x - x3
        mul_lanes(Y3, L, T);  // lambda*(x - x3)
        sub_lanes(Y3, Y3, Y); // - y
        store5(x + FE8 * g, X3);
        store5(y + FE8 * g, Y3);
    }
}

#endif // BACKEND_AVX512IFMA
