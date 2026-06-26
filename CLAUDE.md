# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Three independent SIMD stages select their backend by build tag:** the
RIPEMD160 hash (`ripemd160simd/`, tags `avx2`/`sse4`/none), the secp256k1 field
multiply (`field52simd/`, tag `avx512ifma`/none), and the SHA-256 of the pubkey
(`sha256simd/`, tag `shani`/none). The **accelerated build combines all three
tags** (`avx2 avx512ifma shani`). Pick the backends for the target CPU and build
on that machine:

```bash
# ACCELERATED (Tiger Lake dev machine): AVX2 RIPEMD160 + AVX-512 IFMA field mul.
# GOAMD64=v3 also enables AVX2/BMI2 for the Go glue. cgo required.
GOAMD64=v3 CGO_ENABLED=1 go build -tags "avx2 avx512ifma shani" -o btcpuzzle-ifma .

# AVX2 only: AVX2 RIPEMD160 but the field math falls back to PURE GO (slow).
# Use only on AVX2-without-AVX512-IFMA CPUs; it is NOT the fast path on Tiger Lake.
GOAMD64=v3 CGO_ENABLED=1 go build -tags avx2 -o btcpuzzle-avx2 .

# SSE4 (older x86-64, no AVX2; cgo). 4-way RIPEMD160; field math pure Go.
GOAMD64=v2 CGO_ENABLED=1 go build -tags sse4 -o btcpuzzle-sse4 .

# Pure-Go fallback (no tag, no cgo) — builds/runs anywhere; slow, for correctness.
CGO_ENABLED=0 go build -o btcpuzzle-purego .

# Run
./btcpuzzle-ifma

# Tests / bench / profile — pass the SAME tags to exercise those backends.
GOAMD64=v3 CGO_ENABLED=1 go test -tags "avx2 avx512ifma shani" ./...
CGO_ENABLED=0 go test ./...                      # tests the pure-Go fallbacks
GOAMD64=v3 CGO_ENABLED=1 go test -tags "avx2 avx512ifma shani" -run TestName ./...
GOAMD64=v3 CGO_ENABLED=1 go test -tags "avx2 avx512ifma shani" -run=xxx -bench=BatchedIncremental -benchtime=3s -count=3 .
GOAMD64=v3 CGO_ENABLED=1 go test -tags "avx2 avx512ifma shani" -run=xxx -bench=BatchedIncremental -cpuprofile=/tmp/cpu.prof . && go tool pprof -top /tmp/cpu.prof

# Tidy dependencies
go mod tidy
```

The `avx2`/`sse4`/`avx512ifma` backends require **cgo** (`gcc`/`clang` on PATH,
`CGO_ENABLED=1`) because they compile SIMD C. The **no-tag** build is pure Go and
needs no C compiler. Two tag rules matter: (1) `field52simd`'s `.c` carries a
`//go:build avx512ifma` constraint (unlike `ripemd160simd`'s), so under cgo it
must be paired with `avx512ifma` or it stays pure Go; the main package therefore
only builds two ways under cgo — both stage tags, or none with `CGO_ENABLED=0`.
(2) Build/test commands must carry the right tags; running without them silently
exercises the pure-Go fallback instead of the SIMD path — and a build with just
`-tags avx2` leaves the **field multiply in pure Go**, i.e. slower than this
project's own scalar-btcec history.

The program must be run from the project root directory, as it opens data files relative to the working directory (`data/wallets.json`, `data/ranges.json`, `data/hash160s.json`).

**Build environment note:** this binary targets **only x86-64 Linux with AVX2**, developed and benchmarked on an **Intel Core i7-1185G7 (11th Gen, Tiger Lake; AVX2 + AVX-512 F/DQ/BW/VL/IFMA + SHA-NI)**. No Windows / cross-compile requirement. Build with a native Linux Go toolchain and `GOAMD64=v3`. `GOAMD64=v4` (auto-AVX-512) is still no faster — the Go compiler doesn't auto-vectorize the scalar code — but AVX-512 now pays off via the **hand-written IFMA path** in `field52simd` (`-tags avx512ifma`, requires `avx512ifma` + `avx512f`/`avx512vl`). The required ISA is AVX2 + the rest of the `v3` set (BMI2/FMA/F16C) for the base build, plus AVX-512 F/VL/IFMA for the accelerated field kernel; SHA-NI is used for SHA-256 if present. This is the AVX2 variant — CPU-specific variants for other processors are separate repos (`github.com/daniborgss/btcpuzzle-<target>`). The old committed `bitcoin_finder.exe` Windows binary has been removed; build `bitcoin_finder` instead. Build artifacts (`/bitcoin_finder`, `*.exe`, `*.test`, `*.prof`) and `found_key_*.txt` (which contain **private keys**) are git-ignored. Benchmarks on this laptop are noisy (thermal/background load) — take the **minimum** ns/op across `-count=N` runs as the signal.

## Purpose

This is a CLI tool for participating in the **Bitcoin Puzzle challenges** (the well-known 1000 BTC / 32 BTC puzzle transactions, where keys for puzzles 1–160 are deliberately constrained to known ranges and solving them is the intended goal). The 160 wallets and their key ranges in `data/` correspond to these puzzle entries.

**This code exists solely for the puzzle challenge. It must never be adapted or used to attack regular wallets that don't belong to the puzzle.**

## Architecture

The tool searches private keys within a puzzle's defined range to find the key matching the target puzzle wallet's Hash160 (RIPEMD-160 of SHA-256 of the compressed public key).

### Flow

1. `main.go` — entry point: loads data, picks the wallet number (1–160) from the `-wallet` flag or an interactive prompt, resolves the target Hash160 and key range, then calls `searchForPrivateKey` (passing `-workers`, the parallel worker count; 0 = one per logical CPU). Run multiple instances on distinct wallets by splitting cores with `-workers` (sum ≤ logical CPUs) and optionally `taskset`; stacking instances gives no extra throughput.
2. `search.go` — parallel search engine. See **Search algorithm** below. Progress is logged every 10 seconds. On match, writes the private key to `found_key_<hash160prefix>.txt`.
3. `bitcoin.go` — cryptographic primitives: `privateKeyToHash160` (the reference/seed path) and `privateKeyToAddress` (unused in the main path). Uses `btcsuite/btcd` for secp256k1 and `btcutil.Hash160`.
4. `data.go` — data loading: reads `data/hash160s.json` as the primary source; falls back to `data/wallets.json` (address strings) but conversion is not implemented.
5. `models.go` — JSON structs (`WalletData`, `RangeData`, `Range`, `Hash160Data`).
6. `colors.go` — ANSI terminal color constants.
7. `ripemd160simd/` — multi-message RIPEMD160 for fixed 32-byte inputs, with
   **build-tag-selected backends** all exposing the same API (`const Lanes` +
   `HashBatch(out *[Lanes][20]byte, in *[Lanes][32]byte)`): `avx2.go`+`ripemd160_avx2.c`
   (`-tags avx2`, 8-way AVX2 cgo, `Lanes=8`), `sse4.go`+`ripemd160_sse4.c`
   (`-tags sse4`, 4-way SSE2/SSE4 cgo, `Lanes=4`), and `purego.go` (no tag, pure-Go
   fallback, no cgo, `Lanes=8`). `doc.go` documents the contract; the C files are
   guarded by `#ifdef BACKEND_*` (cgo compiles every `.c` regardless of Go tags).
   Tags are mutually exclusive (`sse4` is `sse4 && !avx2`), so a stray double tag
   still resolves to one backend.
   `ripemd160simd_test.go` is backend-agnostic — it checks whatever backend the tags
   select byte-for-byte against `golang.org/x/crypto/ripemd160`.
8. `field52simd/` — 8-way secp256k1 **field arithmetic** in radix 2^52 (5 limbs of
   52 bits; the layout AVX-512 IFMA `vpmadd52` needs). `Fe` is one element; `Fe8`
   (`[5][8]uint64`, memory-identical to `[5]__m512i`) is the 8-lane SoA the kernel
   consumes. The hot ops `MulBatch`/`SqrBatch` are **backend-selected**: `batch_ifma.go`
   + `field52_ifma.c` (`-tags avx512ifma`, cgo, `vpmadd52luq/huq`) or `batch_purego.go`
   (no tag, scalar per-lane fallback). `AddBatch`/`SubBatch`/`InverseFe8` (Fermat
   chain) are shared pure Go over `Fe8`; `purego.go` holds the single-lane scalar
   reference (`Mul`/`Sqr`/`reduceSolinas` + the `reduceCanonical`/`Bytes` fast pack)
   and `transpose.go` the radix-2^26↔2^52 base change and `PackLanes`/`UnpackLanes`.
   The layered differential tests validate every op + the kernel byte-for-byte
   against `btcec`/`big.Int` (`Camada 1`), and `search_test.go` validates the whole
   integrated pipeline against the scalar reference. `field52_ifma.c` carries a
   `//go:build avx512ifma` constraint so it is not an orphan `.c` under `-tags avx2`.
9. `sha256simd/` — 8-way SHA-256 of the fixed 33-byte compressed pubkey, same
   backend pattern: `sha256_shani.c`+`shani.go` (`-tags shani`, the SHA-NI
   extension — one hash at a time, single padded block, 8 lanes per cgo call) or
   `purego.go` (no tag, `crypto/sha256.Sum256`). API `const Lanes=8` +
   `HashBatch(out *[Lanes][32]byte, in *[Lanes][33]byte)`, checked byte-for-byte
   against `crypto/sha256`. `-msha` isn't on cgo's CFLAGS allowlist, so the
   feature is enabled via `__attribute__((target("sha,sse4.1,ssse3")))` in the C,
   not a `-m` flag; the `.c` has a `//go:build shani` constraint (orphan guard).

### Data files (`data/`)

- `hash160s.json` — array of hex-encoded Hash160 values, one per wallet (preferred input)
- `wallets.json` — array of Bitcoin mainnet addresses (used only if `hash160s.json` is absent; conversion path is a stub)
- `ranges.json` — array of `{min, max, status}` objects with 0x-prefixed hex bounds; index aligns 1-to-1 with `hash160s.json`

### `temp/hash160_generator.go`

A standalone utility (its own `main` package) that converts `data/wallets.json` → `data/hash160s.json`. Run it from the `temp/` directory; it resolves paths relative to its parent directory.

### Search algorithm (`search.go`)

The search avoids a full secp256k1 scalar multiplication per key (tens of × faster per key than the naive reference). The design, in `laneSet`:

- **Affine incremental point addition.** Consecutive keys differ by 1, so consecutive public keys differ by `+G`. Each worker holds a batch of `lanesPerWorker` (1024) points kept in **affine** coordinates and advances all of them by one affine point addition per step (`advance`) instead of recomputing from scratch. Affine addition costs ~4 field mults + 1 square per key, versus ~13 + 4 for the old Jacobian-add-plus-separate-conversion path. Only the initial seed points use a full scalar multiplication (`newLaneSet`). **All `advance` field math now runs through the `field52simd` 8-way kernel** (radix-2^52 `Fe8` groups), not scalar `btcec.FieldVal`.
- **Batched (Montgomery) inversion.** Affine addition needs one modular inverse per step (of the slope denominator `x − xG`). `advance` runs the Montgomery trick as **8 parallel chains** (one per SIMD lane position) over the `ng` groups, folded through a single `field52simd.InverseFe8` (Fermat `a^(p-2)` via the kernel) — amortizing the inversions over all 1024 keys. Because the points are already affine, `forEachHash` needs **no** inversion — it unpacks the SoA groups to flat per-lane coords and hashes them, canonicalizing X/Y with `Fe.reduceCanonical` (`Bytes`, no `big.Int`).
- **8-way SIMD hashing.** `forEachHash` gathers live lanes into groups of `ripemd160simd.Lanes`: each lane's SHA256 is computed one at a time (hardware SHA-NI via the stdlib), its 32-byte output staged into `hash160er.shaIn`, and the group's RIPEMD160 is then computed in a single multi-message call (`ripemd160simd.HashBatch`). An incomplete final group (or one shrunk by `dead` lanes) is padded with a copy of the last input; padded lanes are never inspected.
- **Degenerate cases in `advance`.** A lane whose current point equals `G` (e.g. seed key 1) needs point *doubling* (`λ = 3x²/2y`); a lane equal to `−G` sums to the point at infinity and is marked `dead`. Both make the slope denominator 0, which would zero the shared Montgomery product — and a field has no zero divisors, so they are detected **cheaply**: a zero lane in the accumulated product (`anyLaneZero`) triggers a scalar fixup pass (`fixupDegenerate`, using `btcec` for the doubling slope) before the inversion. The common case (no zero lane) never unpacks. `dead`/padding lanes keep `denom = 1` and are skipped when hashing.
- **Random independent lane starts.** Every lane (`numWorkers × lanesPerWorker` of them) gets its own uniformly-random starting key drawn from the full `[minKey, maxKey]` range and marches forward from there, so each run samples genuinely random regions rather than starting near `minKey`. A per-lane `maxTick` bounds the walk to the range (relevant only for small ranges; large puzzle ranges are effectively unbounded and run until a match). Batch and worker counts shrink automatically for ranges smaller than the lane count.
- **btcec is now only for setup, not the hot loop.** `btcec/v2` (re-exporting `secp256k1/v4`'s `JacobianPoint`/`FieldVal`/`ModNScalar`/`ScalarBaseMultNonConst`) seeds the points (`newLaneSet`) and computes the rare doubling slope. All per-key field arithmetic lives in `field52simd` (radix 2^52, limbs < 2^52 feeding `vpmadd52`; see its `doc.go`/`CLAUDE`-style contract).

**Performance history & current bottleneck.** Compounding rewrites over the original Jacobian-per-key loop: (1) affine rewrite (~2×); (2) 8-way AVX2 RIPEMD160 took hashing to a few percent (~3×); (3) **AVX-512 IFMA 8-way field multiply** (`field52simd`, `vpmadd52`) integrated into `advance`; (4) **cgo-call fusion** — the bottleneck after (3) was not SIMD but cgo call *overhead* (`advance` issued ~1800 cgo calls per 1024-key step). Fusing each phase into one call over the whole laneSet (`SlopeSetup`/`MontForward`/`InverseFe8`/`MontBackward`/`PointAdd`, ~5 calls/step), dropping `big.Int` from the canonical pack, and finally vectorizing the canonicalization itself in C (`CanonBytes`, two calls/`forEachHash`) took it from **~518 (pre-integration scalar btcec) → ~314 → ~191 → ~175 ns/key, ~3.0× end to end** (`-tags "avx2 avx512ifma shani"`). **The loop is now SHA-256-bound:** profiling shows ~40% field math (the IFMA C kernel — opaque to pprof, shows up as `runtime.cgocall`), ~23% SHA-256 (`blockSHANI` plus the stdlib `Digest` `Reset`/`Write`/`Sum` framework overhead). (5) **SHA-NI 8-way kernel** (`sha256simd`): the per-lane stdlib hashing framework (Reset/Write/Sum on `crypto/sha256.Digest`) was ~17% of the loop — `sha256.Sum256` shaved ~7% (no state clone) and a hand-written SHA-NI kernel (single padded block, 8 lanes/cgo call) took the rest, **~12% over the stdlib path in a controlled interleaved A/B** (~189 → ~167 ns/key). With both the field math and the SHA framework removed, what's left is the irreducible work: the IFMA field multiply, the SHA-NI and AVX2 hash compressions, and the canonical pack — there is no further single lever. **Don't** chase the field multiply (Amdahl: it's a minority slice). Lesson: at this point cgo *granularity* matters more than per-op SIMD; vectorizing a cheap op as its own cgo call is a wash. Benchmarks on this laptop are noisy — take the **minimum** ns/op across runs, re-profile (`-bench=BatchedIncremental -cpuprofile`) with the **full tag set**, and remember pprof cannot see inside the C kernel (its time is lumped into `runtime.cgocall`). `GOAMD64=v3` helps the Go glue for free.

**Correctness is verified by `search_test.go`** (`TestLaneSetMatchesReference` + `TestLaneSetMultiGroup`), which assert the batched/incremental pipeline yields byte-identical hash160s to the reference `privateKeyToHash160` across multiple keys and advance steps — including the seed-key-1 doubling and, in the multi-group test, cross-group inversion and dead padding lanes. Run it after any change to the field/curve math: silent math bugs produce wrong hashes, not crashes (`CGO_ENABLED=0 go test .` for the pure-Go pipeline, `-tags "avx2 avx512ifma shani"` for the kernel). The `field52simd` differential harness validates each field op + the kernel byte-for-byte against `btcec`/`big.Int`. `bench_test.go` compares the pipeline against the naive full-scalar-mult reference.

### Other design notes

- Coordination: workers share an `atomic.Bool` (`found`) for the fast stop-check, a mutex-guarded result, and a `sync.Once`-closed `doneCh`.
- `walletNum` input is 1-indexed; it's converted to 0-indexed before array access.
