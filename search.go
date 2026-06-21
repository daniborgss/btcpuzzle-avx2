package main

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"hash"
	"math/big"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/btcsuite/btcd/btcec/v2"

	"github.com/daniborgss/btcpuzzle-avx2/ripemd160simd"
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
// SHA256 is done one lane at a time (it's hardware-accelerated via SHA-NI); its
// outputs are buffered and the RIPEMD160 stage runs 8 lanes at once through the
// AVX2 multi-message implementation in ripemd160simd. Buffers are reused so the
// hot loop performs no per-key allocations.
type hash160er struct {
	sha    hash.Hash
	pubkey [33]byte
	shaIn  [ripemd160simd.Lanes][32]byte // SHA256 outputs feeding the RIPEMD160 stage
	h160   [ripemd160simd.Lanes][20]byte // RIPEMD160 outputs
}

func newHash160er() *hash160er {
	return &hash160er{sha: sha256.New()}
}

// sha256Pubkey writes SHA256(0x02/0x03 || x) into dst. x must be normalized.
func (h *hash160er) sha256Pubkey(oddY bool, x *btcec.FieldVal, dst *[32]byte) {
	if oddY {
		h.pubkey[0] = 0x03
	} else {
		h.pubkey[0] = 0x02
	}
	x.PutBytesUnchecked(h.pubkey[1:])

	h.sha.Reset()
	h.sha.Write(h.pubkey[:])
	h.sha.Sum(dst[:0])
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
type laneSet struct {
	x []btcec.FieldVal // affine X, normalized
	y []btcec.FieldVal // affine Y, normalized
	// dead[j] marks a lane that reached the point at infinity (P + G == O,
	// i.e. P == -G). This is a ~2^-256 event in a real search and never
	// happens in practice; the flag just keeps the math well-defined.
	dead []bool
	// Scratch reused across advance() calls to avoid per-step allocation.
	num    []btcec.FieldVal // numerator of the slope lambda
	denom  []btcec.FieldVal // denominator of the slope lambda
	inv    []btcec.FieldVal // 1/denom, from the batched inversion
	prefix []btcec.FieldVal // running prefix products for Montgomery's trick
	xsub   []btcec.FieldVal // second x to subtract in x3 = lambda^2 - x - xsub
}

// newLaneSet seeds one affine point per base key with a full scalar
// multiplication. This is the only place full scalar mults happen; every
// advance() afterwards is a single affine addition per lane.
func newLaneSet(baseKeys []*big.Int) *laneSet {
	n := len(baseKeys)
	ls := &laneSet{
		x:      make([]btcec.FieldVal, n),
		y:      make([]btcec.FieldVal, n),
		dead:   make([]bool, n),
		num:    make([]btcec.FieldVal, n),
		denom:  make([]btcec.FieldVal, n),
		inv:    make([]btcec.FieldVal, n),
		prefix: make([]btcec.FieldVal, n),
		xsub:   make([]btcec.FieldVal, n),
	}
	var k btcec.ModNScalar
	var p btcec.JacobianPoint
	for j, key := range baseKeys {
		k.SetByteSlice(padPrivateKey(key.Bytes(), 32))
		btcec.ScalarBaseMultNonConst(&k, &p)
		p.ToAffine()
		ls.x[j].Set(&p.X)
		ls.x[j].Normalize()
		ls.y[j].Set(&p.Y)
		ls.y[j].Normalize()
	}
	return ls
}

// forEachHash hashes every (already affine) point in the batch and invokes fn
// with each lane's hash160. It needs no field inversion. Live lanes are gathered
// into groups of ripemd160simd.Lanes (8): each lane's SHA256 is computed, then
// the whole group's RIPEMD160 is computed in one AVX2 multi-message call. It
// stops early and returns true if fn returns true.
func (ls *laneSet) forEachHash(h *hash160er, fn func(lane int, h160 []byte) bool) bool {
	const w = ripemd160simd.Lanes
	n := len(ls.x)
	var idx [w]int
	j := 0
	for j < n {
		// Gather up to w live lanes and stage their SHA256 outputs.
		cnt := 0
		for j < n && cnt < w {
			if !ls.dead[j] {
				idx[cnt] = j
				h.sha256Pubkey(ls.y[j].IsOdd(), &ls.x[j], &h.shaIn[cnt])
				cnt++
			}
			j++
		}
		if cnt == 0 {
			continue
		}
		// Pad an incomplete final group with a copy so Hash8 is well-defined;
		// padded lanes are never inspected.
		for k := cnt; k < w; k++ {
			h.shaIn[k] = h.shaIn[cnt-1]
		}
		ripemd160simd.Hash8(&h.h160, &h.shaIn)
		for k := 0; k < cnt; k++ {
			if fn(idx[k], h.h160[k][:]) {
				return true
			}
		}
	}
	return false
}

// advance adds G to every point in the batch using affine point addition,
// sharing a single field inversion across the whole batch (Montgomery's trick).
//
// For a normal lane P=(x,y) with P != +/-G the new point is
//
//	lambda = (y - yG) / (x - xG)
//	x3     = lambda^2 - xG - x
//	y3     = lambda*(x - x3) - y
//
// The denominator (x - xG) is the only field inverse needed, so all lanes'
// denominators are inverted together. A lane where x == xG is the degenerate
// case: if y == yG it is a point doubling (denominator 2y, numerator 3x^2);
// otherwise P == -G and P+G is the point at infinity, which marks the lane dead.
func (ls *laneSet) advance(g *btcec.JacobianPoint) {
	n := len(ls.x)

	var negXg, negYg btcec.FieldVal
	negXg.NegateVal(&g.X, 1)
	negYg.NegateVal(&g.Y, 1)

	// First pass: build per-lane numerator/denominator and the x value to
	// subtract in the x3 formula. denom is forced non-zero for every lane so
	// the shared product (and its inverse) stays well-defined.
	for j := 0; j < n; j++ {
		if ls.dead[j] {
			ls.denom[j].SetInt(1)
			continue
		}
		var dx btcec.FieldVal
		dx.Set(&ls.x[j])
		dx.Add(&negXg)
		dx.Normalize()
		if dx.IsZero() {
			var dy btcec.FieldVal
			dy.Set(&ls.y[j])
			dy.Add(&negYg)
			dy.Normalize()
			if dy.IsZero() {
				// Point doubling: lambda = 3x^2 / 2y.
				ls.num[j].SquareVal(&ls.x[j])
				ls.num[j].MulInt(3)
				ls.denom[j].Set(&ls.y[j])
				ls.denom[j].MulInt(2)
				ls.xsub[j].Set(&ls.x[j])
			} else {
				// P == -G: the sum is the point at infinity.
				ls.dead[j] = true
				ls.denom[j].SetInt(1)
			}
			continue
		}
		// Normal addition.
		ls.num[j].Set(&ls.y[j])
		ls.num[j].Add(&negYg)
		ls.denom[j].Set(&dx)
		ls.xsub[j].Set(&g.X)
	}

	// Batched inversion: inv[j] = 1/denom[j] using one Inverse() for all lanes.
	var acc btcec.FieldVal
	acc.SetInt(1)
	for j := 0; j < n; j++ {
		ls.prefix[j].Set(&acc)
		acc.Mul(&ls.denom[j])
	}
	acc.Normalize()
	acc.Inverse()
	for j := n - 1; j >= 0; j-- {
		ls.inv[j].Mul2(&acc, &ls.prefix[j])
		acc.Mul(&ls.denom[j])
	}

	// Second pass: compute lambda and the new affine point per lane. All temps
	// are declared once outside the loop so the hot path allocates nothing.
	var lambda, x3, y3, t, neg btcec.FieldVal
	for j := 0; j < n; j++ {
		if ls.dead[j] {
			continue
		}
		lambda.Mul2(&ls.num[j], &ls.inv[j])

		// x3 = lambda^2 - x - xsub
		x3.SquareVal(&lambda)
		neg.NegateVal(&ls.x[j], 1)
		x3.Add(&neg)
		neg.NegateVal(&ls.xsub[j], 1)
		x3.Add(&neg)
		x3.Normalize()

		// y3 = lambda*(x - x3) - y
		neg.NegateVal(&x3, 1)
		t.Set(&ls.x[j])
		t.Add(&neg)
		y3.Mul2(&lambda, &t)
		neg.NegateVal(&ls.y[j], 1)
		y3.Add(&neg)
		y3.Normalize()

		ls.x[j].Set(&x3)
		ls.y[j].Set(&y3)
	}
}

// searchForPrivateKey searches for a private key whose compressed public key
// hashes to targetHash160, within [minKey, maxKey], using one goroutine per CPU
// core. Each worker walks a contiguous slice of the range from a shared random
// offset, advancing a batch of lanes with incremental point addition.
func searchForPrivateKey(minKey, maxKey *big.Int, targetHash160 []byte) {
	numWorkers := runtime.NumCPU()
	if numWorkers < 1 {
		numWorkers = 1
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
