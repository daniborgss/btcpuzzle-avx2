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

The RIPEMD160 backend is chosen at build time (see [Compilation](#compilation)),
so the hardware requirement depends on which backend you build:

| Backend (tag) | Requires | Notes |
|---|---|---|
| **`avx2`** | x86-64 with **AVX2** + the `GOAMD64=v3` set (BMI2/FMA/F16C); **SHA-NI** used if present | The primary, fastest build. Any Intel Haswell (2013+) or AMD Zen (2017+) qualifies. |
| **`sse4`** | x86-64 with **SSE2/SSE4** (essentially any x86-64) | For pre-AVX2 CPUs (e.g. Ivy/Sandy Bridge). 4-way. Build with `GOAMD64=v2`. |
| **none (pure-Go)** | nothing special; no C compiler | Universal fallback, slow. |

**Developed & benchmarked on:** Intel Core i7-1185G7 — 11th Gen "Tiger Lake",
4C/8T, AVX2 + AVX-512 (F/DQ/BW/VL/IFMA) + SHA-NI. AVX-512 is present but not used
yet (an AVX-512 IFMA field-arithmetic path is the planned next step).

To check a CPU on Linux:

```
grep -m1 'model name' /proc/cpuinfo
grep -m1 -o 'avx2' /proc/cpuinfo   # prints "avx2" if the avx2 backend will run
```

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

- Go 1.21 or higher
- For the SIMD backends (`avx2`/`sse4`): a C compiler (`gcc`/`clang`) and
  `CGO_ENABLED=1`, plus a CPU that supports the backend's ISA
  (see [Target CPU / hardware](#target-cpu--hardware)). The pure-Go fallback needs
  neither.

## Usage

The program reads its data files relative to the working directory, so run it from
the project root (where the `data/` directory lives):

```
./btcpuzzle-avx2
```

Then enter a wallet number between 1 and 160 when prompted. The search uses one
worker per CPU core and runs until a match is found.

## Compilation

The RIPEMD160 stage has **swappable SIMD backends selected by a build tag**, so one
codebase produces an optimal binary per CPU. Build on the target machine with the
matching tag:

| Target CPU | Build command | Backend |
|---|---|---|
| **x86-64 with AVX2** (this dev machine: Tiger Lake) | `GOAMD64=v3 CGO_ENABLED=1 go build -tags avx2 -o btcpuzzle-avx2 .` | 8-way AVX2 (cgo) |
| **older x86-64, no AVX2** (e.g. Ivy/Sandy Bridge) | `GOAMD64=v2 CGO_ENABLED=1 go build -tags sse4 -o btcpuzzle-sse4 .` | 4-way SSE (cgo) |
| **any platform / no C compiler** | `CGO_ENABLED=0 go build -o btcpuzzle-purego .` | pure-Go fallback |

The `avx2`/`sse4` builds need cgo (`gcc`/`clang`, `CGO_ENABLED=1`); the no-tag build
is pure Go. `GOAMD64=v3` additionally enables AVX2/BMI2 for the Go field arithmetic
(~1.2x). (`GOAMD64=v4`/AVX-512 was measured no faster — the Go compiler doesn't
auto-vectorize the scalar field/hash code, so anything above `v3` only helps via a
hand-written SIMD path like the RIPEMD160 backends.)

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
- **Multi-message SIMD RIPEMD160.** `forEachHash()` gathers lanes into groups of
  `ripemd160simd.Lanes`, runs each lane's SHA256 (hardware SHA-NI), then computes
  the group's RIPEMD160 in one multi-message backend call (`ripemd160simd.HashBatch`).
  The backend is chosen by build tag (8-way AVX2, 4-way SSE4, or pure-Go); on AVX2
  this took hashing from the largest single cost down to a few percent of the loop
  (~3x on the whole search).
- **Random independent lane starts.** Every lane gets its own uniformly-random
  starting key drawn from the full `[minKey, maxKey]` range and marches forward
  from there, so each run samples genuinely random regions of the range.

Together these are roughly a **6x** throughput improvement over the original
scalar-multiply-per-key loop. The dominant remaining cost is the secp256k1 field
arithmetic.

Correctness is verified by `search_test.go` (the affine + SIMD pipeline yields
byte-identical Hash160s to the reference single-key path, including the point-
doubling case) and `ripemd160simd/ripemd160simd_test.go` (each backend matches the
pure-Go reference). Pass the same `-tags` you build with so the tests exercise that
backend:

```
CGO_ENABLED=1 go test -tags avx2 ./...   # AVX2 backend
go test ./...                            # pure-Go fallback
```

`bench_test.go` compares throughput against the naive full-scalar-mult reference.

## Notes

- Large ranges run until a match is found; there is no fixed iteration limit.
- Only active ranges (status=1) are processed.
- The 160 wallets and ranges in `data/` correspond to the puzzle entries.

## License & attribution

This project is an optimized, CPU-targeted derivative of
[lmajowka/btcgoai](https://github.com/lmajowka/btcgoai).

- The modifications contributed here (the affine search rewrite, the
  `ripemd160simd` AVX2 package, and the build/optimization/documentation changes)
  are licensed under the [MIT License](LICENSE) by Daniel Borges.
- The upstream `lmajowka/btcgoai` repository has **no license**, so its original
  code remains all-rights-reserved by its author. The MIT grant here covers only
  the modifications, **not** the underlying original work. See [NOTICE](NOTICE).

If you want to use or redistribute the project as a whole, please confirm the
upstream terms with the original author first.
