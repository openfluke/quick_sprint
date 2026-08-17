package sprint

import "fmt"

const (
	Batch   = 8
	Classes = 4
	TrainN  = 256
	ValN    = 64
)

// Spec describes one layer sprint: a tiny task that plays to that layer.
type Spec struct {
	Name      string
	Strength  string
	Shape     []int // input shape without batch
	Kind      string // blob | sine1 | spatial | volume | seq | tokens | latent1 | latent2 | latent3
	NeedHead  bool   // false for kmeans (cluster probs = classes)
}

func AllSpecs() []Spec {
	return []Spec{
		{Name: "dense", Strength: "class-conditional 8D blobs", Shape: []int{8}, Kind: "blob", NeedHead: true},
		{Name: "cnn1", Strength: "1D sine frequency", Shape: []int{1, 16}, Kind: "sine1", NeedHead: true},
		{Name: "cnn2", Strength: "tiny spatial patterns", Shape: []int{1, 6, 6}, Kind: "spatial", NeedHead: true},
		{Name: "cnn3", Strength: "tiny occupancy volume", Shape: []int{1, 4, 4, 4}, Kind: "volume", NeedHead: true},
		{Name: "convt1", Strength: "1D upsample from a short latent", Shape: []int{2, 4}, Kind: "latent1", NeedHead: true},
		{Name: "convt2", Strength: "2D upsample from a 2×2 latent", Shape: []int{2, 2, 2}, Kind: "latent2", NeedHead: true},
		{Name: "convt3", Strength: "3D upsample from a 2³ latent", Shape: []int{2, 2, 2, 2}, Kind: "latent3", NeedHead: true},
		{Name: "rnn", Strength: "short sequence → last-step class", Shape: []int{8, 4}, Kind: "seq", NeedHead: true},
		{Name: "lstm", Strength: "gated sequence memory", Shape: []int{8, 4}, Kind: "seq", NeedHead: true},
		{Name: "mha", Strength: "token mixing on 8×8", Shape: []int{8, 8}, Kind: "seq", NeedHead: true},
		{Name: "mamba", Strength: "selective SSM on 8×8", Shape: []int{8, 8}, Kind: "seq", NeedHead: true},
		{Name: "gdn", Strength: "gated delta net (float32 honest)", Shape: []int{8, 8}, Kind: "seq", NeedHead: true},
		{Name: "embedding", Strength: "token-id lookup", Shape: []int{8}, Kind: "tokens", NeedHead: true},
		{Name: "swiglu", Strength: "gated FFN on blobs", Shape: []int{8}, Kind: "blob", NeedHead: true},
		{Name: "layernorm", Strength: "affine after a dense stem", Shape: []int{8}, Kind: "blob", NeedHead: true},
		{Name: "rmsnorm", Strength: "RMS affine after a dense stem", Shape: []int{8}, Kind: "blob", NeedHead: true},
		{Name: "softmax", Strength: "Dense→Softmax vs one-hot", Shape: []int{8}, Kind: "blob", NeedHead: false},
		{Name: "kmeans", Strength: "soft clusters = classes", Shape: []int{8}, Kind: "blob", NeedHead: false},
		{Name: "sequential", Strength: "stacked Dim→Dim dense", Shape: []int{8}, Kind: "blob", NeedHead: true},
		{Name: "residual", Strength: "skip + dense F", Shape: []int{8}, Kind: "blob", NeedHead: true},
		{Name: "parallel", Strength: "two-way add (cameral)", Shape: []int{8}, Kind: "blob", NeedHead: true},
		{Name: "metacognition", Strength: "heuristic-wrapped dense", Shape: []int{8}, Kind: "blob", NeedHead: true},
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
