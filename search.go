package main

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"math/big"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/btcsuite/btcd/btcec/v2"

	"github.com/daniborgss/btcpuzzle/field52simd"
	"github.com/daniborgss/btcpuzzle/ripemd160simd"
	"github.com/daniborgss/btcpuzzle/sha256simd"
)

// lanesPerWorker is how many independent keys each worker advances in lockstep.
// A larger batch amortizes the single (expensive) field inversion over more
// keys via Montgomery's trick, at the cost of more memory per worker.
const lanesPerWorker = 1024

// generatorPoint returns the secp256k1 generator G as an affine (Z=1) point
// with normalized coordinates, ready to use as the constant addend in advance().
func generatorPoint() btcec.JacobianPoint {
	var one btcec.ModNScalar
	one.SetInt(1)
	var g btcec.JacobianPoint
	btcec.ScalarBaseMultNonConst(&one, &g)
	g.ToAffine()
	g.X.Normalize()
	g.Y.Normalize()
	return g
}

// hash160er computes RIPEMD160(SHA256(compressedPubKey)) for a batch of lanes.
// Both stages run through multi-message backends selected by build tag: the 8
// compressed pubkeys are SHA256'd in one call (SHA-NI or stdlib), then their
// digests feed the RIPEMD160 stage in groups of ripemd160simd.Lanes. Buffers are
// reused so the hot loop performs no per-key allocations.
type hash160er struct {
	msgs  [sha256simd.Lanes][33]byte // compressed pubkeys (0x02/03 || X)
	shaIn [sha256simd.Lanes][32]byte // SHA256 outputs feeding RIPEMD160
	h160  [ripemd160simd.Lanes][20]byte
}

func newHash160er() *hash160er {
	return &hash160er{}
}

// laneSet holds a batch of curve points kept in affine coordinates and advanced
// together. Because the points are already affine, forEachHash() needs no field
// inversion at all — it hashes the stored x/y directly. The single (expensive)
// field inversion happens once per advance() step, where the affine point
// addition for the whole batch shares one Montgomery-batched inversion.
//
// Using affine addition (instead of Jacobian addition + a separate
// Jacobian->affine conversion) cuts the field multiplications per key from
// ~13 mul + 4 sq down to ~4 mul + 1 sq, which is the dominant cost of the loop.
// Points are stored in the radix-2^52 SoA layout (field52simd.Fe8) the IFMA
// kernel consumes: ng groups of 8 lanes. The affine X/Y and all advance()
// scratch are groups; per-lane scalars (dead flags, the unpacked coords used
// for hashing) are flat arrays of length ng*8.
type laneSet struct {
	n  int // number of live lanes
	ng int // number of 8-lane groups = ceil(n/8)

	x []field52simd.Fe8 // affine X
	y []field52simd.Fe8 // affine Y
	// dead[j] marks a lane at the point at infinity (P == -G), plus the padding
	// lanes (j >= n) in the final group. Real searches never reach -G (~2^-256);
	// the flag keeps the batched math well-defined.
	dead []bool

	// Scratch reused across advance() to avoid per-step allocation.
	num    []field52simd.Fe8 // numerator of the slope lambda
	denom  []field52simd.Fe8 // denominator of the slope lambda
	inv    []field52simd.Fe8 // 1/denom from the batched inversion
	prefix []field52simd.Fe8 // running prefix products for Montgomery's trick
	xsub   []field52simd.Fe8 // second x to subtract in x3 = lambda^2 - x - xsub

	// Canonical 32-byte big-endian X and Y for every lane, filled by CanonBytes
	// at the top of forEachHash (lane j at [j*32 : j*32+32]).
	xb []byte
	yb []byte
}

// newLaneSet seeds one affine point per base key with a full scalar
// multiplication (the only place full scalar mults happen), converts each to
// the radix-2^52 layout via the byte bridge, and packs them into 8-lane groups.
// Padding lanes in the final group are marked dead.
func newLaneSet(baseKeys []*big.Int) *laneSet {
	const w = field52simd.Lanes
	n := len(baseKeys)
	ng := (n + w - 1) / w
	ls := &laneSet{
		n:      n,
		ng:     ng,
		x:      make([]field52simd.Fe8, ng),
		y:      make([]field52simd.Fe8, ng),
		dead:   make([]bool, ng*w),
		num:    make([]field52simd.Fe8, ng),
		denom:  make([]field52simd.Fe8, ng),
		inv:    make([]field52simd.Fe8, ng),
		prefix: make([]field52simd.Fe8, ng),
		xsub:   make([]field52simd.Fe8, ng),
		xb:     make([]byte, ng*w*32),
		yb:     make([]byte, ng*w*32),
	}
	var k btcec.ModNScalar
	var p btcec.JacobianPoint
	for g := 0; g < ng; g++ {
		var xs, ys [w]field52simd.Fe
		for pos := 0; pos < w; pos++ {
			j := g*w + pos
			if j < n {
				k.SetByteSlice(padPrivateKey(baseKeys[j].Bytes(), 32))
				btcec.ScalarBaseMultNonConst(&k, &p)
				p.ToAffine()
				p.X.Normalize()
				p.Y.Normalize()
				xb := *p.X.Bytes()
				yb := *p.Y.Bytes()
				xs[pos].SetBytes(&xb)
				ys[pos].SetBytes(&yb)
			} else {
				ls.dead[j] = true
				xs[pos] = xs[0]
				ys[pos] = ys[0]
			}
		}
		field52simd.PackLanes(&ls.x[g], &xs)
		field52simd.PackLanes(&ls.y[g], &ys)
	}
	return ls
}

// forEachHash hashes every (already affine) point in the batch and invokes fn
// with each lane's hash160. It needs no field inversion. Live lanes are gathered
// into groups of ripemd160simd.Lanes: each lane's SHA256 is computed, then the
// whole group's RIPEMD160 is computed in one multi-message backend call. It
// stops early and returns true if fn returns true.
func (ls *laneSet) forEachHash(h *hash160er, fn func(lane int, h160 []byte) bool) bool {
	const sw = sha256simd.Lanes
	const rw = ripemd160simd.Lanes // divides sw (both 8, or rw=4)

	// Canonicalize all lanes' X/Y to 32-byte big-endian in two fused calls.
	field52simd.CanonBytes(ls.xb, ls.x)
	field52simd.CanonBytes(ls.yb, ls.y)

	n := ls.n
	var idx [sw]int
	j := 0
	for j < n {
		// Gather up to sw live lanes and build their compressed pubkeys.
		cnt := 0
		for j < n && cnt < sw {
			if !ls.dead[j] {
				idx[cnt] = j
				if ls.yb[j*32+31]&1 == 1 {
					h.msgs[cnt][0] = 0x03
				} else {
					h.msgs[cnt][0] = 0x02
				}
				copy(h.msgs[cnt][1:], ls.xb[j*32:j*32+32])
				cnt++
			}
			j++
		}
		if cnt == 0 {
			continue
		}
		// Pad the incomplete group; padded lanes are never inspected.
		for k := cnt; k < sw; k++ {
			h.msgs[k] = h.msgs[cnt-1]
		}
		sha256simd.HashBatch(&h.shaIn, &h.msgs)
		// RIPEMD160 over the SHA digests, rw at a time.
		for base := 0; base < cnt; base += rw {
			rin := (*[rw][32]byte)(unsafe.Pointer(&h.shaIn[base]))
			ripemd160simd.HashBatch(&h.h160, rin)
			for k := 0; k < rw && base+k < cnt; k++ {
				if fn(idx[base+k], h.h160[k][:]) {
					return true
				}
			}
		}
	}
	return false
}

// advance adds G to every point in the batch using affine point addition,
// sharing a single batched inversion across the whole batch (Montgomery's
// trick), with all field arithmetic in the radix-2^52 8-way kernel.
//
// For a normal lane P=(x,y) with P != +/-G the new point is
//
//	lambda = (y - yG) / (x - xG)
//	x3     = lambda^2 - xG - x
//	y3     = lambda*(x - x3) - y
//
// The denominators (x - xG) are inverted together. A lane with x == xG is the
// degenerate case (doubling if y == yG, else dead); these poison the shared
// product, so they are detected — cheaply, via a zero lane in the accumulated
// product — and fixed up on a scalar path before the inversion.
func (ls *laneSet) advance(g *btcec.JacobianPoint) {
	xGb := *g.X.Bytes()
	yGb := *g.Y.Bytes()
	var xGfe, yGfe field52simd.Fe
	xGfe.SetBytes(&xGb)
	yGfe.SetBytes(&yGb)
	var xG8, yG8 field52simd.Fe8
	broadcastFe8(&xG8, &xGfe)
	broadcastFe8(&yG8, &yGfe)

	// First pass: denom = x - xG, num = y - yG (one fused call over all groups).
	field52simd.SlopeSetup(ls.denom, ls.num, ls.x, ls.y, &xG8, &yG8)
	for grp := 0; grp < ls.ng; grp++ {
		ls.xsub[grp] = xG8 // xsub = xG for normal lanes (doubling overrides it)
	}
	// Force every dead/padding lane's denominator to 1 (cheap direct write) so
	// it never zeroes the shared product.
	for j := 0; j < ls.ng*field52simd.Lanes; j++ {
		if ls.dead[j] {
			setLaneOne(&ls.denom[j/field52simd.Lanes], j%field52simd.Lanes)
		}
	}

	// Batched inversion: forward product, invert the 8 lanes, backward scan.
	// A zero lane in the forward product means some non-dead lane had x == xG
	// (denom 0) — the degenerate case; fix it up and redo the forward scan.
	var acc, invAcc field52simd.Fe8
	field52simd.MontForward(ls.prefix, &acc, ls.denom)
	if anyLaneZero(&acc) {
		ls.fixupDegenerate(&yGfe)
		field52simd.MontForward(ls.prefix, &acc, ls.denom)
	}
	field52simd.InverseFe8(&invAcc, &acc)
	field52simd.MontBackward(ls.inv, &invAcc, ls.prefix, ls.denom)

	// Second pass: lambda, x3, y3 (one fused call over all groups). Dead lanes
	// compute harmless garbage and are skipped when hashing.
	field52simd.PointAdd(ls.x, ls.y, ls.num, ls.inv, ls.xsub)
}

// fixupDegenerate handles the rare lanes whose denominator is 0 (x == xG): a
// point doubling (y == yG → num = 3x^2, denom = 2y, xsub = x) or the point at
// infinity (else → mark dead, denom = 1). Only groups containing such a lane
// are unpacked. Uses btcec for the doubling slope on the scalar path.
func (ls *laneSet) fixupDegenerate(yGfe *field52simd.Fe) {
	const w = field52simd.Lanes
	var negYg field52simd.Fe
	negYg.Negate(yGfe)
	var oneB [32]byte
	oneB[31] = 1

	for grp := 0; grp < ls.ng; grp++ {
		var dens [w]field52simd.Fe
		field52simd.UnpackLanes(&dens, &ls.denom[grp])
		has := false
		for pos := 0; pos < w; pos++ {
			if !ls.dead[grp*w+pos] && dens[pos].IsZero() {
				has = true
				break
			}
		}
		if !has {
			continue
		}
		var xs, ys, nums, xsubs [w]field52simd.Fe
		field52simd.UnpackLanes(&xs, &ls.x[grp])
		field52simd.UnpackLanes(&ys, &ls.y[grp])
		field52simd.UnpackLanes(&nums, &ls.num[grp])
		field52simd.UnpackLanes(&xsubs, &ls.xsub[grp])
		for pos := 0; pos < w; pos++ {
			j := grp*w + pos
			if ls.dead[j] || !dens[pos].IsZero() {
				continue
			}
			var dy field52simd.Fe
			dy.Add(&ys[pos], &negYg)
			if dy.IsZero() {
				// Doubling: num = 3x^2, denom = 2y, xsub = x (via btcec).
				xb := xs[pos].Bytes()
				yb := ys[pos].Bytes()
				var xv, yv, numv, denv btcec.FieldVal
				xv.SetBytes(&xb)
				yv.SetBytes(&yb)
				numv.SquareVal(&xv)
				numv.MulInt(3)
				numv.Normalize()
				denv.Set(&yv)
				denv.MulInt(2)
				denv.Normalize()
				nb := *numv.Bytes()
				db := *denv.Bytes()
				nums[pos].SetBytes(&nb)
				dens[pos].SetBytes(&db)
				xsubs[pos] = xs[pos]
			} else {
				ls.dead[j] = true
				dens[pos].SetBytes(&oneB)
			}
		}
		field52simd.PackLanes(&ls.denom[grp], &dens)
		field52simd.PackLanes(&ls.num[grp], &nums)
		field52simd.PackLanes(&ls.xsub[grp], &xsubs)
	}
}

// broadcastFe8 sets every lane of out to fe.
func broadcastFe8(out *field52simd.Fe8, fe *field52simd.Fe) {
	var a [field52simd.Lanes]field52simd.Fe
	for l := range a {
		a[l] = *fe
	}
	field52simd.PackLanes(out, &a)
}

// setLaneOne sets a single lane of f to the field element 1.
func setLaneOne(f *field52simd.Fe8, pos int) {
	f[0][pos] = 1
	for k := 1; k < 5; k++ {
		f[k][pos] = 0
	}
}

// anyLaneZero reports whether any of the 8 lanes of f is 0 mod p.
func anyLaneZero(f *field52simd.Fe8) bool {
	var fe [field52simd.Lanes]field52simd.Fe
	field52simd.UnpackLanes(&fe, f)
	for l := 0; l < field52simd.Lanes; l++ {
		if fe[l].IsZero() {
			return true
		}
	}
	return false
}

// searchForPrivateKey searches for a private key whose compressed public key
// hashes to targetHash160, within [minKey, maxKey], using one goroutine per CPU
// core. Each worker walks a contiguous slice of the range from a shared random
// offset, advancing a batch of lanes with incremental point addition.
func searchForPrivateKey(minKey, maxKey *big.Int, targetHash160 []byte, workers int) {
	// workers <= 0 means "one per logical CPU"; otherwise use the requested count
	// (e.g. to split a machine across several instances; see the -workers flag).
	numWorkers := workers
	if numWorkers < 1 {
		numWorkers = runtime.NumCPU()
	}

	// Inclusive count of keys in the range.
	rangeLen := new(big.Int).Sub(maxKey, minKey)
	rangeLen.Add(rangeLen, big.NewInt(1))

	// Shrink the batch (then the worker count) so we never have more lanes than
	// keys in the range (only relevant for tiny puzzle ranges).
	lanes := lanesPerWorker
	totalLanes := func(l, w int) *big.Int {
		return new(big.Int).Mul(big.NewInt(int64(w)), big.NewInt(int64(l)))
	}
	for lanes > 1 && totalLanes(lanes, numWorkers).Cmp(rangeLen) > 0 {
		lanes /= 2
	}
	for numWorkers > 1 && totalLanes(lanes, numWorkers).Cmp(rangeLen) > 0 {
		numWorkers--
	}
	nLanes := numWorkers * lanes

	// Give every lane its own uniformly-random starting key drawn from the whole
	// [minKey, maxKey] range; each lane then advances forward from its own point.
	// This makes every run sample genuinely random regions of the range instead
	// of always beginning near minKey.
	bases := make([]*big.Int, nLanes)
	for i := range bases {
		off, err := rand.Int(rand.Reader, rangeLen)
		if err != nil {
			off = big.NewInt(0)
		}
		bases[i] = off.Add(off, minKey)
	}

	// maxTickWithinRange reports how many forward steps a lane may take before
	// leaving the range, or -1 when that exceeds int64 (i.e. effectively
	// unbounded for any real run — the normal case for large puzzle ranges).
	maxTickWithinRange := func(base *big.Int) int64 {
		d := new(big.Int).Sub(maxKey, base)
		if d.IsInt64() {
			return d.Int64()
		}
		return -1
	}

	fmt.Printf("%sStarting key search with %d workers x %d lanes...%s\n", ColorBlue, numWorkers, lanes, ColorReset)
	firstLaneBase := bases[0]
	fmt.Printf("%sRandom start point (worker 0, lane 0): %s%s%s\n", ColorCyan, ColorBoldCyan, hex.EncodeToString(firstLaneBase.Bytes()), ColorReset)

	// Coordination state.
	var (
		found     atomic.Bool
		totalIter int64
		lastTick  atomic.Int64
		resultMu  sync.Mutex
		foundKey  []byte
		foundH    []byte
		wg        sync.WaitGroup
		closeOnce sync.Once
		doneCh    = make(chan struct{})
	)
	signalDone := func() { closeOnce.Do(func() { close(doneCh) }) }

	startTime := time.Now()

	// Progress reporter.
	go func() {
		for !found.Load() {
			time.Sleep(10 * time.Second)
			if found.Load() {
				return
			}
			it := atomic.LoadInt64(&totalIter)
			kps := float64(it) / time.Since(startTime).Seconds()
			lastKey := new(big.Int).Add(firstLaneBase, big.NewInt(lastTick.Load()))
			fmt.Printf("%sChecked %d keys (%.2f keys/sec) - Last key: %s%s\n",
				ColorCyan, it, kps, hex.EncodeToString(lastKey.Bytes()), ColorReset)
		}
	}()

	// Workers.
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()

			baseKeys := bases[w*lanes : (w+1)*lanes]
			maxTicks := make([]int64, lanes)
			unbounded := false
			var maxTick int64
			for j, b := range baseKeys {
				mt := maxTickWithinRange(b)
				maxTicks[j] = mt
				if mt < 0 {
					unbounded = true
				} else if mt > maxTick {
					maxTick = mt
				}
			}

			ls := newLaneSet(baseKeys)
			h := newHash160er()
			g := generatorPoint()

			for tick := int64(0); ; tick++ {
				if found.Load() {
					return
				}

				matched := ls.forEachHash(h, func(lane int, h160 []byte) bool {
					if maxTicks[lane] >= 0 && tick > maxTicks[lane] {
						return false // this lane has walked past maxKey
					}
					if !bytes.Equal(h160, targetHash160) {
						return false
					}
					key := new(big.Int).Add(baseKeys[lane], big.NewInt(tick))
					resultMu.Lock()
					if !found.Load() {
						found.Store(true)
						foundKey = padPrivateKey(key.Bytes(), 32)
						foundH = append([]byte(nil), h160...)
					}
					resultMu.Unlock()
					signalDone()
					return true
				})

				atomic.AddInt64(&totalIter, int64(lanes))
				if matched {
					return
				}
				if w == 0 {
					lastTick.Store(tick)
				}

				// Stop once every lane has walked past maxKey. Only reachable for
				// small ranges; large puzzle ranges run until a match is found.
				if !unbounded && tick >= maxTick {
					return
				}
				ls.advance(&g)
			}
		}(w)
	}

	// Signal completion when all workers exit without a match.
	go func() {
		wg.Wait()
		signalDone()
	}()

	<-doneCh

	// Report results.
	resultMu.Lock()
	defer resultMu.Unlock()
	if found.Load() {
		privateKeyHex := hex.EncodeToString(foundKey)
		hash160Hex := hex.EncodeToString(foundH)
		fmt.Printf("\n%sMATCH FOUND!%s\n", ColorBoldGreen, ColorReset)
		fmt.Printf("%sPrivate Key: %s%s%s\n", ColorGreen, ColorBoldGreen, privateKeyHex, ColorReset)
		fmt.Printf("%sHash160: %s%s%s\n", ColorGreen, ColorBoldGreen, hash160Hex, ColorReset)

		filename := "found_key_" + hash160Hex[:8] + ".txt"
		content := fmt.Sprintf("Private Key: %s\nHash160: %s\nFound at: %s", privateKeyHex, hash160Hex, time.Now().Format(time.RFC3339))
		if err := os.WriteFile(filename, []byte(content), 0600); err != nil {
			fmt.Printf("%sError writing key to file: %s%s\n", ColorRed, err, ColorReset)
		} else {
			fmt.Printf("%sPrivate key saved to file: %s%s%s\n", ColorGreen, ColorBoldGreen, filename, ColorReset)
		}
	} else {
		fmt.Printf("\n%sNo match found after checking approximately %d keys.%s\n", ColorYellow, atomic.LoadInt64(&totalIter), ColorReset)
	}
}
