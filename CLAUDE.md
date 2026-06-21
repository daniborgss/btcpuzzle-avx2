# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The RIPEMD160 stage has **swappable backends selected by build tag** (see
`ripemd160simd/`). Pick the backend for the target CPU and build on that machine:

```bash
# AVX2 (x86-64 w/ AVX2; cgo). GOAMD64=v3 also enables AVX2/BMI2 for the Go field
# math (~1.2x). This is the build for the Tiger Lake dev machine.
GOAMD64=v3 CGO_ENABLED=1 go build -tags avx2 -o btcpuzzle-avx2 .

# SSE4 (older x86-64, no AVX2; cgo). 4-way backend.
GOAMD64=v2 CGO_ENABLED=1 go build -tags sse4 -o btcpuzzle-sse4 .

# Pure-Go fallback (no tag, no cgo) — builds/runs anywhere; slow, for correctness.
CGO_ENABLED=0 go build -o btcpuzzle-purego .

# Run
./btcpuzzle-avx2

# Tests / bench / profile — pass the SAME -tags to exercise that backend
CGO_ENABLED=1 go test -tags avx2 ./...
go test ./...                                   # tests the pure-Go fallback
CGO_ENABLED=1 go test -tags avx2 -run TestName ./...
CGO_ENABLED=1 go test -tags avx2 -run=xxx -bench=BatchedIncremental -benchtime=3s -count=3 .
CGO_ENABLED=1 go test -tags avx2 -run=xxx -bench=BatchedIncremental -cpuprofile=/tmp/cpu.prof . && go tool pprof -top /tmp/cpu.prof

# Tidy dependencies
go mod tidy
```

The `avx2`/`sse4` backends require **cgo** (`gcc`/`clang` on PATH, `CGO_ENABLED=1`)
because `ripemd160simd` compiles SIMD C. The **no-tag** build is pure Go and needs
no C compiler. Build commands must carry the right `-tags`; running tests/benches
without the tag silently exercises the pure-Go fallback instead of the SIMD path.

The program must be run from the project root directory, as it opens data files relative to the working directory (`data/wallets.json`, `data/ranges.json`, `data/hash160s.json`).

**Build environment note:** this binary targets **only x86-64 Linux with AVX2**, developed and benchmarked on an **Intel Core i7-1185G7 (11th Gen, Tiger Lake; AVX2 + AVX-512 F/DQ/BW/VL/IFMA + SHA-NI)**. No Windows / cross-compile requirement. Build with a native Linux Go toolchain and `GOAMD64=v3` (`GOAMD64=v4`/AVX-512 measured no faster because the Go compiler doesn't auto-vectorize the scalar field/hash code — AVX-512 would only pay off via a hand-written SIMD path). The required ISA is AVX2 + the rest of the `v3` set (BMI2/FMA/F16C); SHA-NI is used for SHA-256 if present. This is the AVX2 variant — CPU-specific variants for other processors are separate repos (`github.com/daniborgss/btcpuzzle-<target>`). The old committed `bitcoin_finder.exe` Windows binary has been removed; build `bitcoin_finder` instead. Build artifacts (`/bitcoin_finder`, `*.exe`, `*.test`, `*.prof`) and `found_key_*.txt` (which contain **private keys**) are git-ignored. Benchmarks on this laptop are noisy (thermal/background load) — take the **minimum** ns/op across `-count=N` runs as the signal.

## Purpose

This is a CLI tool for participating in the **Bitcoin Puzzle challenges** (the well-known 1000 BTC / 32 BTC puzzle transactions, where keys for puzzles 1–160 are deliberately constrained to known ranges and solving them is the intended goal). The 160 wallets and their key ranges in `data/` correspond to these puzzle entries.

**This code exists solely for the puzzle challenge. It must never be adapted or used to attack regular wallets that don't belong to the puzzle.**

## Architecture

The tool searches private keys within a puzzle's defined range to find the key matching the target puzzle wallet's Hash160 (RIPEMD-160 of SHA-256 of the compressed public key).

### Flow

1. `main.go` — entry point: loads data, prompts for wallet number (1–160), resolves the target Hash160 and key range, then calls `searchForPrivateKey`.
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

### Data files (`data/`)

- `hash160s.json` — array of hex-encoded Hash160 values, one per wallet (preferred input)
- `wallets.json` — array of Bitcoin mainnet addresses (used only if `hash160s.json` is absent; conversion path is a stub)
- `ranges.json` — array of `{min, max, status}` objects with 0x-prefixed hex bounds; index aligns 1-to-1 with `hash160s.json`

### `temp/hash160_generator.go`

A standalone utility (its own `main` package) that converts `data/wallets.json` → `data/hash160s.json`. Run it from the `temp/` directory; it resolves paths relative to its parent directory.

### Search algorithm (`search.go`)

The search avoids a full secp256k1 scalar multiplication per key (tens of × faster per key than the naive reference). The design, in `laneSet`:

- **Affine incremental point addition.** Consecutive keys differ by 1, so consecutive public keys differ by `+G`. Each worker holds a batch of `lanesPerWorker` (1024) points kept in **affine** coordinates and advances all of them by one affine point addition per step (`advance`) instead of recomputing from scratch. Affine addition costs ~4 field mults + 1 square per key, versus ~13 + 4 for the old Jacobian-add-plus-separate-conversion path. Only the initial seed points use a full scalar multiplication (`newLaneSet`).
- **Batched (Montgomery) inversion.** Affine addition needs one modular inverse per step (of the slope denominator `x − xG`, the expensive field op). `advance` inverts the whole batch's denominators with a single `FieldVal.Inverse()` plus multiplications, amortizing one inversion over 1024 keys. Because the points are already affine, `forEachHash` needs **no** inversion — it hashes the stored coordinates directly (X written via `PutBytesUnchecked`, no per-key allocation).
- **8-way SIMD hashing.** `forEachHash` gathers live lanes into groups of 8 (`ripemd160simd.Lanes`): each lane's SHA256 is computed one at a time (hardware SHA-NI via the stdlib), its 32-byte output staged into `hash160er.shaIn`, and the group's RIPEMD160 is then computed in a single AVX2 multi-message call (`ripemd160simd.Hash8`). An incomplete final group (or one shrunk by `dead` lanes) is padded with a copy of the last input; padded lanes are never inspected. This replaced the per-lane stdlib RIPEMD160 and measured ~3× on the whole loop (the pure-Go hasher's per-call `Reset`/`Write`/`Sum` overhead was larger than profiles attributed to its compression).
- **Degenerate cases in `advance`.** A lane whose current point equals `G` (e.g. seed key 1) needs point *doubling* (`λ = 3x²/2y`); the formula switches denominator/numerator for that lane. A lane equal to `−G` would sum to the point at infinity and is marked `dead` (a ~2⁻²⁵⁶ event, never hit in a real run). `dead` lanes keep `denom = 1` so they don't poison the shared product, and are skipped when hashing.
- **Random independent lane starts.** Every lane (`numWorkers × lanesPerWorker` of them) gets its own uniformly-random starting key drawn from the full `[minKey, maxKey]` range and marches forward from there, so each run samples genuinely random regions rather than starting near `minKey`. A per-lane `maxTick` bounds the walk to the range (relevant only for small ranges; large puzzle ranges are effectively unbounded and run until a match). Batch and worker counts shrink automatically for ranges smaller than the lane count.
- These primitives come from `btcec/v2`, which re-exports `secp256k1/v4`'s `JacobianPoint`, `FieldVal`, `ModNScalar`, and `ScalarBaseMultNonConst`. All `FieldVal` ops require magnitude ≤ 8 and output magnitude 1, so chained `Mul`/`Square`/`Add`/`NegateVal` must stay within that budget before a `Normalize`; `Normalize` before `Bytes`/`PutBytesUnchecked`/`IsOdd`/`IsZero`.

**Performance history & current bottleneck.** Two compounding rewrites, ~6× total over the original Jacobian-per-key loop: (1) the affine rewrite halved the EC field work (~2×); (2) the 8-way AVX2 RIPEMD160 took hashing from ~41% of the loop to a few percent (~3× more). **After both, the dominant cost is the secp256k1 field arithmetic (`FieldVal.Mul2` in `advance`), now ~⅔ of the loop**; SHA256 (~10%) is next; RIPEMD160 is no longer significant. The next real lever is therefore the EC math, e.g. an AVX-512 **IFMA** (`vpmadd52*`) vectorized field multiply / point addition — a large effort that would make lanes SIMD-native end to end. Making RIPEMD160 wider (AVX-512 16-way) is **not** worth it: it's already a small slice (Amdahl). Benchmarks on this laptop are noisy — take the **minimum** ns/op across runs, and re-profile (`CGO_ENABLED=1 go test -bench=BatchedIncremental -cpuprofile`) before acting on bottleneck assumptions. `GOAMD64=v3` helps the field arithmetic for free.

**Correctness is verified by `search_test.go`**, which asserts the batched/incremental pipeline yields byte-identical hash160s to the reference `privateKeyToHash160` across multiple keys and advance steps (including the seed-key-1 doubling case). Run it (`go test ./...`) after any change to the field/curve math — silent math bugs produce wrong hashes, not crashes. `bench_test.go` compares the affine pipeline against the naive full-scalar-mult reference.

### Other design notes

- Coordination: workers share an `atomic.Bool` (`found`) for the fast stop-check, a mutex-guarded result, and a `sync.Once`-closed `doneCh`.
- `walletNum` input is 1-indexed; it's converted to 0-indexed before array access.
