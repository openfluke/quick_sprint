package sprint

import (
	"math"
	"math/rand/v2"
	"sync"

	"github.com/openfluke/tide/runner"
	"github.com/openfluke/welvet/core"
)

type synthDS struct {
	mu     sync.Mutex
	spec   Spec
	train  []item
	val    []item
	batch  int
	seed   uint64
	rng    *rand.Rand
	offset int
	order  []int
	phase  string
}

type item struct {
	lab  int
	seed uint64
}

func newSynth(spec Spec, trainN, valN, batch int, seed uint64) *synthDS {
	if batch < 1 {
		batch = Batch
	}
	if trainN < batch {
		trainN = batch
	}
	trainN = (trainN / batch) * batch
	valN = (valN / batch) * batch
	if valN < batch {
		valN = batch
	}
	d := &synthDS{
		spec:  spec,
		batch: batch,
		seed:  seed,
		rng:   rand.New(rand.NewPCG(seed, seed^0xC0FFEE)),
		phase: "A",
	}
	d.train = makeItems(trainN, seed^0x51A7)
	d.val = makeItems(valN, seed^0x7A1D)
	d.ResetEpoch(0)
	return d
}

func makeItems(n int, seed uint64) []item {
	out := make([]item, n)
	for i := 0; i < n; i++ {
		out[i] = item{lab: i % Classes, seed: seed + uint64(i)*0x9E3779B97F4A7C15}
	}
	return out
}

func shapeElems(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func phaseSeed(phase string) uint64 {
	switch phase {
	case "B":
		return 2
	case "A2":
		return 3
	default:
		return 1
	}
}

func (d *synthDS) SetPhase(phase string) {
	d.mu.Lock()
	d.phase = phase
	d.mu.Unlock()
}

func fill(spec Spec, x []float32, lab int, rng *rand.Rand, phase string) {
	switch spec.Kind {
	case "sine1":
		fillSine(x, lab, rng, phase)
	case "spatial":
		fillSpatial(spec, x, lab, rng, phase)
	case "volume":
		fillVolume(spec, x, lab, rng, phase)
	case "delay":
		fillDelay(spec, x, lab, rng, phase)
	case "assoc":
		fillAssoc(spec, x, lab, rng, phase)
	case "tokens":
		fillTokens(x, lab, rng, phase)
	case "latent":
		fillXOR(x, lab, rng, phase)
	case "xor-scale":
		fillXOR(x, lab, rng, phase)
		scale := 0.35 + rng.Float32()*2.4
		for i := range x {
			x[i] *= scale
		}
	default: // xor
		fillXOR(x, lab, rng, phase)
	}
}

func bitF(b int) float32 {
	if b == 1 {
		return 1
	}
	return 0
}

// fillXOR is 4-class pairwise XOR: label = (x0⊕x1)*2 + (x2⊕x3). Linear
// readout of the four bits cannot solve it. Phase B moves the bits.
func fillXOR(x []float32, lab int, rng *rand.Rand, phase string) {
	for i := range x {
		x[i] = float32(rng.NormFloat64() * 0.4)
	}
	y0 := (lab >> 1) & 1
	y1 := lab & 1
	a, c := rng.IntN(2), rng.IntN(2)
	b, d := a^y0, c^y1
	off := 0
	if phase == "B" && len(x) >= 8 {
		off = 4
	}
	if phase == "A2" && len(x) >= 6 {
		off = 2
	}
	if len(x) < off+4 {
		off = 0
	}
	if len(x) >= off+4 {
		x[off+0] = bitF(a) + float32(rng.NormFloat64()*0.08)
		x[off+1] = bitF(b) + float32(rng.NormFloat64()*0.08)
		x[off+2] = bitF(c) + float32(rng.NormFloat64()*0.08)
		x[off+3] = bitF(d) + float32(rng.NormFloat64()*0.08)
	}
}

func fillSine(x []float32, lab int, rng *rand.Rand, phase string) {
	freq := float64(lab + 1)
	switch phase {
	case "B":
		freq += 2
	case "A2":
		freq += 1
	}
	for i := range x {
		x[i] = float32(math.Sin(freq*float64(i)*0.45+rng.Float64()*0.2)) + float32(rng.NormFloat64()*0.25)
	}
}

func rotRC(r, c, h, w, k int) (int, int) {
	switch k & 3 {
	case 1:
		return c, h - 1 - r
	case 2:
		return h - 1 - r, w - 1 - c
	case 3:
		return w - 1 - c, r
	default:
		return r, c
	}
}

func fillSpatial(spec Spec, x []float32, lab int, rng *rand.Rand, phase string) {
	h, w := spec.Shape[1], spec.Shape[2]
	k := 0
	if phase == "B" {
		k = 1
	} else if phase == "A2" {
		k = 2
	}
	for i := range x {
		rr, cc := rotRC(i/w, i%w, h, w, k)
		v := float32(rng.NormFloat64() * 0.35)
		switch lab {
		case 0:
			if rr == h/4 || rr == 3*h/4 {
				v += 1.1
			}
		case 1:
			if cc == w/4 || cc == 3*w/4 {
				v += 1.1
			}
		case 2:
			if rr == cc || rr+cc == h-1 {
				v += 1.1
			}
		default:
			if (rr-h/2)*(rr-h/2)+(cc-w/2)*(cc-w/2) <= 2 {
				v += 1.1
			}
		}
		x[i] = v
	}
}

func fillVolume(spec Spec, x []float32, lab int, rng *rand.Rand, phase string) {
	d, h, w := spec.Shape[1], spec.Shape[2], spec.Shape[3]
	k := 0
	if phase == "B" {
		k = 1
	} else if phase == "A2" {
		k = 2
	}
	for i := range x {
		z := i / (h * w)
		r := (i / w) % h
		c := i % w
		if k == 1 {
			z, r = r, d-1-z
		} else if k == 2 {
			r, c = c, h-1-r
		}
		v := float32(rng.NormFloat64() * 0.3)
		switch lab {
		case 0:
			if z == 0 || z == d-1 {
				v += 1
			}
		case 1:
			if r == 0 || r == h-1 {
				v += 1
			}
		case 2:
			if c == 0 || c == w-1 {
				v += 1
			}
		default:
			if z == r && r == c {
				v += 1
			}
		}
		x[i] = v
	}
}

func fillDelay(spec Spec, x []float32, lab int, rng *rand.Rand, phase string) {
	t, dim := spec.Shape[0], spec.Shape[1]
	cue := 0
	switch phase {
	case "B":
		cue = t - 1
	case "A2":
		cue = t / 2
	}
	for i := 0; i < t; i++ {
		for j := 0; j < dim; j++ {
			x[i*dim+j] = float32(rng.NormFloat64() * 0.25)
		}
		if i == cue {
			x[i*dim+(lab%dim)] += 1.6
		}
	}
}

func fillAssoc(spec Spec, x []float32, lab int, rng *rand.Rand, phase string) {
	t, dim := spec.Shape[0], spec.Shape[1]
	if dim < 8 {
		fillXOR(x, lab, rng, phase)
		return
	}
	for i := range x {
		x[i] = float32(rng.NormFloat64() * 0.08)
	}
	q := rng.IntN(Classes)
	if phase == "B" {
		q = (q + 1) % Classes
	}
	putOH := func(tok, dimOff, cls int) {
		base := tok*dim + dimOff
		for c := 0; c < Classes && base+c < len(x); c++ {
			if c == cls {
				x[base+c] = 1
			} else {
				x[base+c] = 0
			}
		}
	}
	putOH(0, 0, q)
	matchAt := 1
	if t > 2 {
		matchAt = 1 + rng.IntN(t-1)
	}
	if phase == "A2" {
		matchAt = t - 1
	}
	for tok := 1; tok < t; tok++ {
		key := rng.IntN(Classes)
		val := rng.IntN(Classes)
		if tok == matchAt {
			key, val = q, lab
		} else if key == q {
			key = (key + 1) % Classes
		}
		putOH(tok, 0, key)
		putOH(tok, 4, val)
	}
}

func fillTokens(x []float32, lab int, rng *rand.Rand, phase string) {
	const vocab = 16
	a := rng.IntN(vocab)
	b := rng.IntN(vocab)
	for (a^b)%Classes != lab {
		b = rng.IntN(vocab)
	}
	if phase == "B" {
		a, b = b, a
	}
	x[0], x[1] = float32(a), float32(b)
	for i := 2; i < len(x); i++ {
		x[i] = float32(rng.IntN(vocab))
	}
}

func (d *synthDS) NextServe(phase string) runner.Sample {
	d.mu.Lock()
	defer d.mu.Unlock()
	if phase == "" {
		phase = d.phase
	}
	return d.pack(d.val, d.rng, phase)
}

func (d *synthDS) TrainLen() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.train)
}

func (d *synthDS) EpochOffset() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.offset
}

func (d *synthDS) ResetEpoch(offset int) {
	d.mu.Lock()
	defer d.mu.Unlock()
	n := len(d.train)
	d.order = make([]int, n)
	for i := range d.order {
		d.order[i] = i
	}
	shuf := rand.New(rand.NewPCG(d.seed, d.seed^0xE90C4))
	for i := n - 1; i > 0; i-- {
		j := shuf.IntN(i + 1)
		d.order[i], d.order[j] = d.order[j], d.order[i]
	}
	if offset < 0 {
		offset = 0
	}
	if offset > n {
		offset = n
	}
	d.offset = offset
}

func (d *synthDS) NextTrain() (runner.Sample, bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.offset >= len(d.train) {
		return runner.Sample{}, false
	}
	end := d.offset + d.batch
	if end > len(d.train) {
		end = len(d.train)
	}
	batch := make([]item, 0, end-d.offset)
	for i := d.offset; i < end; i++ {
		batch = append(batch, d.train[d.order[i]])
	}
	d.offset = end
	return d.packItems(batch, d.phase), true
}

func (d *synthDS) pack(pool []item, rng *rand.Rand, phase string) runner.Sample {
	batch := make([]item, d.batch)
	for i := range batch {
		batch[i] = pool[rng.IntN(len(pool))]
	}
	return d.packItems(batch, phase)
}

func (d *synthDS) packItems(batch []item, phase string) runner.Sample {
	b := len(batch)
	shape := append([]int{b}, d.spec.Shape...)
	x := core.NewTensor[float32](shape...)
	target := core.NewTensor[float32](b, Classes)
	labels := make([]int, b)
	n := shapeElems(d.spec.Shape)
	for i, it := range batch {
		rng := rand.New(rand.NewPCG(it.seed, phaseSeed(phase)))
		fill(d.spec, x.Data[i*n:(i+1)*n], it.lab, rng, phase)
		labels[i] = it.lab
		target.Data[i*Classes+it.lab] = 1
	}
	return runner.Sample{X: x, Target: target, Labels: labels}
}
