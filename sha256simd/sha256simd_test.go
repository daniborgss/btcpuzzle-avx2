package sha256simd

import (
	"crypto/rand"
	"crypto/sha256"
	"testing"
)

// TestHashBatch checks the selected backend byte-for-byte against the standard
// library over random 33-byte messages, plus boundary inputs.
func TestHashBatch(t *testing.T) {
	check := func(in *[Lanes][33]byte) {
		t.Helper()
		var out [Lanes][32]byte
		HashBatch(&out, in)
		for l := 0; l < Lanes; l++ {
			if want := sha256.Sum256(in[l][:]); out[l] != want {
				t.Fatalf("lane %d msg %x:\n got  %x\n want %x", l, in[l], out[l], want)
			}
		}
	}

	// Edge inputs: all-zero, all-0xff, and the compressed-pubkey prefixes.
	var edge [Lanes][33]byte
	for i := range edge[1] {
		edge[1][i] = 0xff
	}
	edge[2][0] = 0x02
	edge[3][0] = 0x03
	for i := 1; i < 33; i++ {
		edge[3][i] = byte(i)
	}
	check(&edge)

	for iter := 0; iter < 50000; iter++ {
		var in [Lanes][33]byte
		for l := range in {
			if _, err := rand.Read(in[l][:]); err != nil {
				t.Fatal(err)
			}
		}
		check(&in)
	}
}
