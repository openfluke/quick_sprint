package sprint

import "fmt"

const (
	Batch   = 8
	Classes = 4
	TrainN  = 1024
	ValN    = 128
	CellMS  = 2000 // Lucy-style wall clock per cell (test48 perm race is 2s)
)

// Spec describes one layer sprint: a toy that actually needs that mid Op.
type Spec struct {
	Name     string
	Strength string
	Shape    []int  // input shape without batch
	Kind     string // xor | xor-scale | sine1 | spatial | volume | delay | assoc | tokens | latent
	NeedHead bool   // false for kmeans (cluster probs = classes)
}

func AllSpecs() []Spec {
	return []Spec{
		{Name: "dense", Strength: "4-class pairwise XOR in noise", Shape: []int{8}, Kind: "xor", NeedHead: true},
		{Name: "cnn1", Strength: "1D sine freq class, Lucy 1→2→3 switches", Shape: []int{1, 16}, Kind: "sine1", NeedHead: true},
		{Name: "cnn2", Strength: "tiny spatial patterns (phase rotates)", Shape: []int{1, 8, 8}, Kind: "spatial", NeedHead: true},
		{Name: "cnn3", Strength: "tiny occupancy volume (phase rotates)", Shape: []int{1, 4, 4, 4}, Kind: "volume", NeedHead: true},
		{Name: "convt1", Strength: "1D upsample from XOR latent", Shape: []int{2, 4}, Kind: "latent", NeedHead: true},
		{Name: "convt2", Strength: "2D upsample from XOR latent", Shape: []int{2, 2, 2}, Kind: "latent", NeedHead: true},
		{Name: "convt3", Strength: "3D upsample from XOR latent", Shape: []int{2, 2, 2, 2}, Kind: "latent", NeedHead: true},
		{Name: "rnn", Strength: "class cue at t=0, answer at last step", Shape: []int{8, 4}, Kind: "delay", NeedHead: true},
		{Name: "lstm", Strength: "gated delay: cue early, ignore the rest", Shape: []int{8, 4}, Kind: "delay", NeedHead: true},
		{Name: "mha", Strength: "associative recall: match query key", Shape: []int{8, 8}, Kind: "assoc", NeedHead: true},
		{Name: "mamba", Strength: "selective SSM associative recall", Shape: []int{8, 8}, Kind: "assoc", NeedHead: true},
		{Name: "gdn", Strength: "gated delta associative recall (f32)", Shape: []int{8, 8}, Kind: "assoc", NeedHead: true},
		{Name: "embedding", Strength: "token-id XOR of first two ids", Shape: []int{8}, Kind: "tokens", NeedHead: true},
		{Name: "swiglu", Strength: "gated FFN on pairwise XOR", Shape: []int{8}, Kind: "xor", NeedHead: true},
		{Name: "layernorm", Strength: "XOR after random per-sample scale", Shape: []int{8}, Kind: "xor-scale", NeedHead: true},
		{Name: "rmsnorm", Strength: "XOR after random per-sample scale", Shape: []int{8}, Kind: "xor-scale", NeedHead: true},
		{Name: "softmax", Strength: "overlapping XOR → softmax vs one-hot", Shape: []int{8}, Kind: "xor", NeedHead: false},
		{Name: "kmeans", Strength: "overlapping XOR clusters", Shape: []int{8}, Kind: "xor", NeedHead: false},
		{Name: "sequential", Strength: "stacked dense on pairwise XOR", Shape: []int{8}, Kind: "xor", NeedHead: true},
		{Name: "residual", Strength: "skip + dense F on pairwise XOR", Shape: []int{8}, Kind: "xor", NeedHead: true},
		{Name: "parallel", Strength: "two-way add on pairwise XOR", Shape: []int{8}, Kind: "xor", NeedHead: true},
		{Name: "metacognition", Strength: "heuristic-wrapped XOR", Shape: []int{8}, Kind: "xor", NeedHead: true},
	}
}

func Lookup(name string) (Spec, error) {
	for _, s := range AllSpecs() {
		if s.Name == name {
			return s, nil
		}
	}
	return Spec{}, fmt.Errorf("unknown layer %q", name)
}

func Names() []string {
	all := AllSpecs()
	out := make([]string, len(all))
	for i, s := range all {
		out[i] = s.Name
	}
	return out
}
