package sprint

import (
	"math"
	"math/rand/v2"
	"sync"

	"github.com/openfluke/tide/runner"
	"github.com/openfluke/welvet/core"
)

type synthDS struct {
	mu      sync.Mutex
	spec    Spec
	train   []item
	val     []item
	batch   int
	seed    uint64
	rng     *rand.Rand
	offset  int
	order   []int
}

type item struct {
	x      []float32
	label  int
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
	}
	gen := rand.New(rand.NewPCG(seed, seed^0x51A7))
	d.train = makeItems(spec, trainN, gen)
	d.val = makeItems(spec, valN, gen)
	d.ResetEpoch(0)
	return d
}

func makeItems(spec Spec, n int, rng *rand.Rand) []item {
	out := make([]item, n)
	for i := 0; i < n; i++ {
		lab := i % Classes
		x := make([]float32, shapeElems(spec.Shape))
		fill(spec, x, lab, rng)
		out[i] = item{x: x, label: lab}
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

func fill(spec Spec, x []float32, lab int, rng *rand.Rand) {
	switch spec.Kind {
	case "sine1":
		// [C=1, L]
		freq := float64(lab+1) * 1.5
		for i := range x {
			x[i] = float32(math.Sin(freq*float64(i)/2 + rng.Float64()*0.15))
		}
	case "spatial":
		h, w := spec.Shape[1], spec.Shape[2]
		for i := range x {
			r, c := i/w, i%w
			switch lab {
			case 0:
				if r < h/2 {
					x[i] = 1
				}
			case 1:
				if c < w/2 {
					x[i] = 1
				}
			case 2:
				if (r+c)%2 == 0 {
					x[i] = 1
				}
			default:
				if (r-h/2)*(r-h/2)+(c-w/2)*(c-w/2) < 4 {
					x[i] = 1
				}
			}
			x[i] += float32(rng.Float64()*0.1 - 0.05)
		}
	case "volume":
		d, h, w := spec.Shape[1], spec.Shape[2], spec.Shape[3]
		for i := range x {
			z := i / (h * w)
			r := (i / w) % h
			c := i % w
			switch lab {
			case 0:
				if z < d/2 {
					x[i] = 1
				}
			case 1:
				if r < h/2 {
					x[i] = 1
				}
			case 2:
				if c < w/2 {
					x[i] = 1
				}
			default:
				x[i] = float32((z + r + c) % 2)
			}
			x[i] += float32(rng.Float64()*0.08 - 0.04)
		}
	case "seq":
		// [T, D] — bump a class-specific channel along the sequence
		t, dim := spec.Shape[0], spec.Shape[1]
		for i := 0; i < t; i++ {
			for j := 0; j < dim; j++ {
				x[i*dim+j] = float32(rng.NormFloat64() * 0.05)
			}
			x[i*dim+(lab%dim)] += 1.2
			if i == lab%t {
				x[i*dim+(lab%dim)] += 0.8
			}
		}
	case "tokens":
		vocab := 16
		for i := range x {
			id := (lab*3 + i) % vocab
			x[i] = float32(id)
		}
	case "latent1", "latent2", "latent3":
		for i := range x {
			x[i] = float32(rng.NormFloat64() * 0.2)
			if i%Classes == lab {
				x[i] += 0.9
			}
		}
	default: // blob
		for i := range x {
			x[i] = float32(rng.NormFloat64() * 0.15)
		}
		centroid := lab % len(x)
		x[centroid] += 1.4
		if len(x) > 4 {
			x[(centroid+4)%len(x)] += 0.6
		}
	}
}

func (d *synthDS) NextServe(phase string) runner.Sample {
	_ = phase
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.pack(d.val, d.rng)
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
	return d.packItems(batch), true
}

func (d *synthDS) pack(pool []item, rng *rand.Rand) runner.Sample {
	batch := make([]item, d.batch)
	for i := range batch {
		batch[i] = pool[rng.IntN(len(pool))]
	}
	return d.packItems(batch)
}

func (d *synthDS) packItems(batch []item) runner.Sample {
	b := len(batch)
	shape := append([]int{b}, d.spec.Shape...)
	x := core.NewTensor[float32](shape...)
	target := core.NewTensor[float32](b, Classes)
	labels := make([]int, b)
	n := shapeElems(d.spec.Shape)
	for i, it := range batch {
		copy(x.Data[i*n:(i+1)*n], it.x)
		labels[i] = it.label
		target.Data[i*Classes+it.label] = 1
	}
	return runner.Sample{X: x, Target: target, Labels: labels}
}
