package sprint

import (
	"math"
	"testing"
)

func TestXORPairwiseNotCentroid(t *testing.T) {
	spec, _ := Lookup("dense")
	ds := newSynth(spec, 32, 32, 8, 99)
	s := ds.NextServe("A")
	n := shapeElems(spec.Shape)
	// A linear "class = argmax dim" centroid toy would put the bump on lab%8.
	// Pairwise XOR writes bits into 4 dims; those dims must not uniquely be lab.
	for b := 0; b < len(s.Labels); b++ {
		lab := s.Labels[b]
		x := s.X.Data[b*n : (b+1)*n]
		if argmax(x) == lab {
			// allowed sometimes by chance; count across batch
		}
		_ = lab
		if math.Abs(float64(x[lab%n])) > 2 && math.Abs(float64(x[(lab+1)%n])) < 0.2 {
			t.Fatalf("looks like the old centroid blob (lab=%d x=%v)", lab, x)
		}
	}
}

func TestSinePhaseSwitches(t *testing.T) {
	spec, _ := Lookup("cnn1")
	ds := newSynth(spec, 8, 8, 8, 7)
	a := ds.NextServe("A")
	b := ds.NextServe("B")
	if sameSlice(a.X.Data, b.X.Data) {
		t.Fatal("sine phase A and B produced identical inputs")
	}
}

func TestDelayCueMovesWithPhase(t *testing.T) {
	spec, _ := Lookup("lstm")
	ds := newSynth(spec, 8, 8, 8, 3)
	tLen, dim := spec.Shape[0], spec.Shape[1]
	energy := func(phase string, timestep int) float64 {
		s := ds.NextServe(phase)
		n := shapeElems(spec.Shape)
		e := 0.0
		for b := 0; b < len(s.Labels); b++ {
			row := s.X.Data[b*n+timestep*dim : b*n+(timestep+1)*dim]
			for _, v := range row {
				e += math.Abs(float64(v))
			}
		}
		return e
	}
	if energy("A", 0) < energy("A", tLen-1) {
		t.Fatal("delay phase A should put the cue at t=0")
	}
	if energy("B", tLen-1) < energy("B", 0) {
		t.Fatal("delay phase B should put the cue at last step")
	}
}

func TestAssocHasQueryAndMatch(t *testing.T) {
	spec, _ := Lookup("mha")
	ds := newSynth(spec, 8, 8, 8, 11)
	s := ds.NextServe("A")
	tLen, dim := spec.Shape[0], spec.Shape[1]
	n := shapeElems(spec.Shape)
	for b := 0; b < len(s.Labels); b++ {
		tok0 := s.X.Data[b*n : b*n+dim]
		q := argmax(tok0[:4])
		found := false
		for tok := 1; tok < tLen; tok++ {
			row := s.X.Data[b*n+tok*dim : b*n+(tok+1)*dim]
			if argmax(row[:4]) == q && argmax(row[4:8]) == s.Labels[b] {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("no matching key for query %d lab %d", q, s.Labels[b])
		}
	}
}

func TestBuildAndStepHarderToys(t *testing.T) {
	cell := testCell()
	for _, name := range []string{"dense", "cnn1", "cnn2", "lstm", "mha", "layernorm", "embedding"} {
		net, err := BuildNet(name, cell)
		if err != nil {
			t.Fatalf("%s build: %v", name, err)
		}
		spec, _ := Lookup(name)
		ds := newSynth(spec, Batch, Batch, Batch, 1)
		s := ds.NextServe("A")
		if _, err := net.TrainStep(s.X, s.Target, 0.05, cell.Mode); err != nil {
			t.Fatalf("%s train: %v", name, err)
		}
		if _, _, err := net.ServeEval(s.X, s.Target); err != nil {
			t.Fatalf("%s serve: %v", name, err)
		}
	}
}

func argmax(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}

func sameSlice(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
