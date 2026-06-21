package ripemd160simd

import (
	"bytes"
	"crypto/rand"
	"testing"

	"golang.org/x/crypto/ripemd160"
)

func refHash(msg []byte) [20]byte {
	h := ripemd160.New()
	h.Write(msg)
	var out [20]byte
	copy(out[:], h.Sum(nil))
	return out
}

// TestHash8MatchesStdlib checks every lane against the pure-Go reference over
// many random batches, plus the edge cases of all-zero and all-0xff inputs.
func TestHash8MatchesStdlib(t *testing.T) {
	var in [Lanes][32]byte
	var out [Lanes][20]byte

	check := func(name string) {
		Hash8(&out, &in)
		for l := 0; l < Lanes; l++ {
			want := refHash(in[l][:])
			if !bytes.Equal(out[l][:], want[:]) {
				t.Fatalf("%s lane %d:\n  got  %x\n  want %x", name, l, out[l], want)
			}
		}
	}

	// Fixed edge cases.
	check("all-zero")
	for l := 0; l < Lanes; l++ {
		for j := range in[l] {
			in[l][j] = 0xff
		}
	}
	check("all-ff")

	// Random batches.
	for iter := 0; iter < 200; iter++ {
		for l := 0; l < Lanes; l++ {
			if _, err := rand.Read(in[l][:]); err != nil {
				t.Fatal(err)
			}
		}
		check("random")
	}
}

// BenchmarkSIMD8 measures throughput of the 8-way AVX2 path (reported per
// message via SetBytes-free accounting: ns/op is per batch of 8).
func BenchmarkSIMD8(b *testing.B) {
	var in [Lanes][32]byte
	var out [Lanes][20]byte
	for l := 0; l < Lanes; l++ {
		rand.Read(in[l][:])
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Hash8(&out, &in)
	}
}

// BenchmarkStdlib8 is the apples-to-apples baseline: eight sequential pure-Go
// RIPEMD160 hashes (what the current search loop does per 8 keys).
func BenchmarkStdlib8(b *testing.B) {
	var in [Lanes][32]byte
	for l := 0; l < Lanes; l++ {
		rand.Read(in[l][:])
	}
	h := ripemd160.New()
	var sum [20]byte
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for l := 0; l < Lanes; l++ {
			h.Reset()
			h.Write(in[l][:])
			h.Sum(sum[:0])
		}
	}
}
