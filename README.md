# Bitcoin Puzzle Finder — AVX2 + AVX-512 IFMA / Intel Tiger Lake build

A CLI tool for the **Bitcoin Puzzle challenges** (the well-known 1000 BTC / 32 BTC
puzzle transactions, where the keys for puzzles 1–160 are deliberately constrained
to known ranges and solving them is the intended goal). It searches the private
keys within a puzzle's defined range to find the one whose compressed public key
hashes to the target wallet's Hash160.

This build is **tuned for x86-64 with AVX2 + AVX-512 IFMA** and was developed and
benchmarked on an **Intel Core i7-1185G7 (11th Gen, Tiger Lake)**. Three SIMD
stages are selected at build time: the RIPEMD160 hash (AVX2), the secp256k1 field
multiply (AVX-512 IFMA, `vpmadd52`), and the SHA-256 of the pubkey (SHA-NI). The
accelerated binary combines all three and runs the EC point arithmetic 8 lanes at
a time — about **3× the throughput** of the earlier scalar build. See [Target CPU / hardware](#target-cpu--hardware) for exact
requirements. CPU-specific variants for other processors live in sibling
repositories.

> This code exists solely for the puzzle challenge. It must never be adapted or
> used against regular wallets that don't belong to the puzzle.

## Target CPU / hardware

The two SIMD stages are chosen at build time (see [Compilation](#compilation)) via
build tags, so the hardware requirement depends on which you build:

| Build (tags) | Requires | Notes |
|---|---|---|
| **`avx2 avx512ifma shani`** | x86-64 with **AVX2**, **AVX-512 F/VL/IFMA**, and **SHA-NI**; `GOAMD64=v3` set | The fastest build (AVX2 RIPEMD160 + IFMA field + SHA-NI). Intel Ice/Tiger Lake+, AMD Zen 4+. |
| **`avx2`** | x86-64 with **AVX2** + the `GOAMD64=v3` set (BMI2/FMA/F16C) | AVX2 RIPEMD160 only; the **field math falls back to pure Go** (slower). For AVX2 CPUs without AVX-512 IFMA. |
| **`sse4`** | x86-64 with **SSE2/SSE4** (essentially any x86-64) | 4-way RIPEMD160; field math pure Go. For pre-AVX2 CPUs. Build with `GOAMD64=v2`. |
| **none (pure-Go)** | nothing special; no C compiler | Universal fallback, slow. |

**Developed & benchmarked on:** Intel Core i7-1185G7 — 11th Gen "Tiger Lake",
4C/8T, AVX2 + AVX-512 (F/DQ/BW/VL/IFMA) + SHA-NI — which supports the full
`avx2 avx512ifma shani` build.

To check a CPU on Linux:

```
grep -m1 'model name' /proc/cpuinfo
grep -m1 -o 'avx512ifma' /proc/cpuinfo   # prints "avx512ifma" if the fast field kernel will run
grep -m1 -o 'avx2' /proc/cpuinfo         # prints "avx2" if the AVX2 hash backend will run
```

## Description

The program works as follows:

1. It takes a wallet number (1–160) — from the `-wallet` flag, or by prompting
   when the flag is omitted.
2. It resolves that wallet's target Hash160 from `data/hash160s.json` and its key
   range from `data/ranges.json`.
3. It searches private keys within that range, computing
   `RIPEMD160(SHA256(compressedPubKey))` for each candidate.
4. When a candidate's Hash160 matches the target, it prints the private key and
   writes it to `found_key_<hash160prefix>.txt`.

Progress (keys checked, keys/sec, last key) is logged every 10 seconds.

## Prerequisites

- Go 1.21 or higher
- For the SIMD backends (`avx2`/`sse4`/`avx512ifma`): a C compiler (`gcc`/`clang`)
  and `CGO_ENABLED=1`, plus a CPU that supports the backend's ISA
  (see [Target CPU / hardware](#target-cpu--hardware)). The pure-Go fallback needs
  neither.

## Usage

The program reads its data files relative to the working directory, so run it from
the project root (where the `data/` directory lives):

```
./btcpuzzle-ifma -wallet 65            # non-interactive: search wallet 65
./btcpuzzle-ifma                       # interactive: prompts for the wallet number
./btcpuzzle-ifma -wallet 65 -workers 4 # limit to 4 parallel workers
```

`-wallet N` selects the puzzle (1–160) without prompting (handy for scripts and
benchmarking); omit it to be prompted. `-workers N` sets how many parallel search
workers to spawn; the default (0) is one per logical CPU. The search runs until a
match is found (Ctrl-C to stop). On a match the private key is printed and written
to `found_key_<hash160prefix>.txt` — **this file contains a private key** and is
git-ignored. Low-numbered wallets solve instantly; high-numbered ones have
astronomically large ranges and run indefinitely.

### Running several instances

One instance already uses every logical CPU, so a single search needs no flags.
To search **different wallets in parallel**, split the machine with `-workers` so
the instances don't oversubscribe the cores (the sum of all `-workers` should not
exceed the logical CPU count — e.g. 8 on a 4-core/8-thread laptop):

```
# Two wallets, sharing all 8 threads (≈ half throughput each):
./btcpuzzle-ifma -wallet 71 -workers 4 &
./btcpuzzle-ifma -wallet 73 -workers 4 &

# Stronger isolation: pin each to its own 2 physical cores (no HT cross-talk,
# avoids cross-instance AVX-512 downclock):
taskset -c 0,1 ./btcpuzzle-ifma -wallet 71 -workers 2 &
taskset -c 2,3 ./btcpuzzle-ifma -wallet 73 -workers 2 &
```

`-workers` caps how many cores each instance grabs; `taskset` pins *which* cores.
Stacking instances on the same wallet/machine gives no extra throughput — they
contend for the same cores — so only run multiple instances for distinct wallets.

## Compilation

Both SIMD stages are **selected by build tags**, so one codebase produces an
optimal binary per CPU. Build on the target machine with the matching tags:

| Target CPU | Build command | Backends |
|---|---|---|
| **AVX2 + AVX-512 IFMA** (this dev machine: Tiger Lake) | `GOAMD64=v3 CGO_ENABLED=1 go build -tags "avx2 avx512ifma shani" -o btcpuzzle-ifma .` | AVX2 RIPEMD160 + IFMA field (cgo) |
| **AVX2, no AVX-512 IFMA** | `GOAMD64=v3 CGO_ENABLED=1 go build -tags avx2 -o btcpuzzle-avx2 .` | AVX2 RIPEMD160; **field math pure Go** |
| **older x86-64, no AVX2** (e.g. Ivy/Sandy Bridge) | `GOAMD64=v2 CGO_ENABLED=1 go build -tags sse4 -o btcpuzzle-sse4 .` | 4-way SSE RIPEMD160; field pure Go |
| **any platform / no C compiler** | `CGO_ENABLED=0 go build -o btcpuzzle-purego .` | pure-Go fallback |

> **Use both tags for the fast build.** `-tags avx2` alone compiles and runs, but
> leaves the secp256k1 field arithmetic in pure Go — *slower* than the previous
> scalar build. The `avx512ifma` tag is what enables the vectorized field multiply.

The cgo builds need a C compiler (`gcc`/`clang`, `CGO_ENABLED=1`); the no-tag build
is pure Go. `GOAMD64=v3` also enables AVX2/BMI2 for the Go glue. (`GOAMD64=v4`/auto
AVX-512 is no faster — the Go compiler doesn't auto-vectorize the scalar code; the
AVX-512 win comes from the hand-written IFMA kernel, not the compiler.)

## How the search works

The search avoids a full secp256k1 scalar multiplication per key. The design lives
in `search.go`:

- **Affine incremental addition.** Consecutive keys differ by 1, so consecutive
  public keys differ by `+G`. Points are kept in **affine coordinates** and each
  `advance()` step adds `G` to every lane with the affine addition formula. This
  needs only ~4 field multiplications + 1 square per key (versus ~13 + 4 for
  Jacobian addition plus a separate Jacobian→affine conversion), which is why
  vectorizing that field arithmetic (below) pays off.
- **8-way AVX-512 IFMA field arithmetic.** The EC point arithmetic runs in the
  `field52simd` kernel: field elements in radix 2^52 (5 limbs), 8 independent
  lanes packed into AVX-512 registers, multiplied with `vpmadd52`. The whole
  `advance()` step is fused into ~5 cgo calls (`SlopeSetup` → `MontForward` →
  `InverseFe8` → `MontBackward` → `PointAdd`) over the whole lane set, so cgo call
  overhead is negligible. Degenerate cases (a lane at ±G: doubling or the point at
  infinity) are detected cheaply (a zero lane in the shared product) and fixed up
  on a scalar path.
- **Batched (Montgomery) inversion.** The one expensive field inverse per
  `advance()` is shared across the whole batch via Montgomery's trick — run as 8
  parallel chains (one per SIMD lane) folded through a single vectorized inverse.
  Because the points stay affine, `forEachHash()` needs no inversion; it
  canonicalizes the coordinates 8 lanes at a time (also in the kernel) and hashes
  them, allocating nothing per key.
- **SIMD hashing.** `forEachHash()` builds the 8 compressed pubkeys and SHA256s
  them in one `sha256simd.HashBatch` call (a SHA-NI kernel that drops the stdlib
  `hash.Hash` framework overhead, or `crypto/sha256` without the tag), then
  computes RIPEMD160 over the digests in one multi-message call
  (`ripemd160simd.HashBatch`, 8-way AVX2 / 4-way SSE4 / pure-Go).
- **Random independent lane starts.** Every lane gets its own uniformly-random
  starting key drawn from the full `[minKey, maxKey]` range and marches forward
  from there, so each run samples genuinely random regions of the range.

Together these are roughly a **3×** throughput improvement from the IFMA field
kernel (and the SHA-NI kernel) on top of the earlier ~6× affine + SIMD-RIPEMD160
rewrite. With the field multiply and the SHA framework both removed, what remains
is the irreducible compute — the IFMA field multiply, the SHA-NI and AVX2 hash
compressions, and the canonical byte pack; the field arithmetic is no longer the
dominant cost.

Correctness is verified by `search_test.go` (`TestLaneSetMatchesReference` and
`TestLaneSetMultiGroup` — the full IFMA pipeline yields byte-identical Hash160s to
the reference single-key path across multiple groups, padding lanes, and the
point-doubling case), the `field52simd` differential tests (every field op + the
kernel checked byte-for-byte against `btcec`/`big.Int`), and
`ripemd160simd/ripemd160simd_test.go`. Pass the same tags you build with:

```
GOAMD64=v3 CGO_ENABLED=1 go test -tags "avx2 avx512ifma shani" ./...   # full SIMD build
CGO_ENABLED=0 go test ./...                                      # pure-Go fallback
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
