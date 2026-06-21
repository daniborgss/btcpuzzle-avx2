# Bitcoin Puzzle Finder — AVX2 / Intel Tiger Lake build

A CLI tool for the **Bitcoin Puzzle challenges** (the well-known 1000 BTC / 32 BTC
puzzle transactions, where the keys for puzzles 1–160 are deliberately constrained
to known ranges and solving them is the intended goal). It searches the private
keys within a puzzle's defined range to find the one whose compressed public key
hashes to the target wallet's Hash160.

This build is **tuned for x86-64 with AVX2** and was developed and benchmarked on
an **Intel Core i7-1185G7 (11th Gen, Tiger Lake)**. See
[Target CPU / hardware](#target-cpu--hardware) for exact requirements. CPU-specific
variants for other processors live in sibling repositories.

> This code exists solely for the puzzle challenge. It must never be adapted or
> used against regular wallets that don't belong to the puzzle.

## Target CPU / hardware

| | |
|---|---|
| **Architecture** | x86-64 (64-bit Intel/AMD) |
| **Required ISA extensions** | **AVX2**, BMI2, FMA, F16C, LZCNT/MOVBE (the `GOAMD64=v3` feature set). The AVX2 RIPEMD160 path will not run without AVX2. |
| **Used if present** | **SHA-NI** (hardware SHA-256) — strongly recommended for full speed |
| **Developed & benchmarked on** | **Intel Core i7-1185G7** — 11th Gen "Tiger Lake", 4C/8T, AVX2 + AVX-512 (F/DQ/BW/VL/IFMA) + SHA-NI |
| **AVX-512** | Present on the dev CPU but **not used** by this build (the SIMD hashing is AVX2 8-way). A future AVX-512 IFMA path is the planned next step. |

To check your CPU on Linux:

```
grep -m1 'model name' /proc/cpuinfo
grep -m1 -o 'avx2' /proc/cpuinfo   # must print "avx2"
```

Any reasonably modern Intel (Haswell, 2013+) or AMD (Zen, 2017+) x86-64 CPU has
AVX2 and can build/run this; the Tiger Lake tuning (and the `GOAMD64=v3` flag) just
reflects where it was measured.

## Description

The program works as follows:

1. It prompts the user to enter a wallet number (1–160).
2. It resolves that wallet's target Hash160 from `data/hash160s.json` and its key
   range from `data/ranges.json`.
3. It searches private keys within that range, computing
   `RIPEMD160(SHA256(compressedPubKey))` for each candidate.
4. When a candidate's Hash160 matches the target, it prints the private key and
   writes it to `found_key_<hash160prefix>.txt`.

Progress (keys checked, keys/sec, last key) is logged every 10 seconds.

## Prerequisites

- An x86-64 CPU with **AVX2** (see [Target CPU / hardware](#target-cpu--hardware))
- Go 1.21 or higher
- A C compiler (`gcc`/`clang`) — the RIPEMD160 stage uses an AVX2 cgo package, so
  builds require `CGO_ENABLED=1` (the default on a native Linux build)

## Usage

The program reads its data files relative to the working directory, so run it from
the project root (where the `data/` directory lives):

```
./bitcoin_finder
```

Then enter a wallet number between 1 and 160 when prompted. The search uses one
worker per CPU core and runs until a match is found.

## Compilation

This binary targets this Linux x86_64 machine only and requires cgo. Build
natively with `GOAMD64=v3` to enable AVX2/BMI2 for the secp256k1 field arithmetic
(~1.2x):

```
GOAMD64=v3 CGO_ENABLED=1 go build -o bitcoin_finder .
```

(`GOAMD64=v4`/AVX-512 was measured to be no faster — the Go compiler doesn't
auto-vectorize the scalar field/hash code, so anything above `v3` only helps via a
hand-written SIMD path like the RIPEMD160 one below.)

## How the search works

The search avoids a full secp256k1 scalar multiplication per key. The design lives
in `search.go`:

- **Affine incremental addition.** Consecutive keys differ by 1, so consecutive
  public keys differ by `+G`. Points are kept in **affine coordinates** and each
  `advance()` step adds `G` to every lane with the affine addition formula. This
  needs only ~4 field multiplications + 1 square per key (versus ~13 + 4 for
  Jacobian addition plus a separate Jacobian→affine conversion), and field
  arithmetic is the dominant cost of the loop.
- **Batched (Montgomery) inversion.** The one expensive field inverse per
  `advance()` is shared across the whole batch of `lanesPerWorker` (1024) lanes:
  all the lanes' denominators are inverted with a single `FieldVal.Inverse()` plus
  multiplications. Because the points are already affine, `forEachHash()` needs no
  inversion at all and hashes the stored coordinates directly, allocating nothing
  per key.
- **8-way SIMD RIPEMD160.** `forEachHash()` gathers lanes into groups of 8, runs
  each lane's SHA256 (hardware SHA-NI), then computes the group's RIPEMD160 in one
  AVX2 multi-message call (`ripemd160simd.Hash8`). This took hashing from the
  largest single cost down to a few percent of the loop (~3x on the whole search).
- **Random independent lane starts.** Every lane gets its own uniformly-random
  starting key drawn from the full `[minKey, maxKey]` range and marches forward
  from there, so each run samples genuinely random regions of the range.

Together these are roughly a **6x** throughput improvement over the original
scalar-multiply-per-key loop. The dominant remaining cost is the secp256k1 field
arithmetic.

Correctness is verified by `search_test.go` (the affine + SIMD pipeline yields
byte-identical Hash160s to the reference single-key path, including the point-
doubling case) and `ripemd160simd/ripemd160simd_test.go` (the AVX2 hasher matches
the pure-Go reference). Run the tests after any change to the field/curve math or
the hashing:

```
CGO_ENABLED=1 go test ./...
```

`bench_test.go` compares throughput against the naive full-scalar-mult reference.

## Notes

- Large ranges run until a match is found; there is no fixed iteration limit.
- Only active ranges (status=1) are processed.
- The 160 wallets and ranges in `data/` correspond to the puzzle entries.
</content>
</invoke>
