package sprint

import (
	"fmt"
	"math"

	"github.com/openfluke/tide/metrics"
	"github.com/openfluke/tide/permute"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/weights"
)

const stepTicks = 3

// StackNet is a tiny Welvet stack that implements runner.Net.
type StackNet struct {
	Stack *parallel.Stack
}

func (n *StackNet) TrainStep(x, target *core.Tensor[float32], lr float64, mode permute.TrainMode) (loss float64, err error) {
	if n == nil || n.Stack == nil {
		return 0, fmt.Errorf("sprint: nil stack")
	}
	wv, err := mode.Welvet()
	if err != nil {
		return 0, err
	}
	ticks := 1
	if mode.IsStepSched() {
		ticks = stepTicks
	}
	for i := 0; i < ticks-1; i++ {
		if _, _, err = parallel.ForwardStack(n.Stack, x); err != nil {
			return 0, err
		}
	}
	return parallel.TrainStackMSE(n.Stack, x, target, wv, lr)
}

func (n *StackNet) ServeEval(x, target *core.Tensor[float32]) (preds []int, softAcc float64, err error) {
	if n == nil || n.Stack == nil {
		return nil, 0, fmt.Errorf("sprint: nil stack")
	}
	_, out, err := parallel.ForwardStack(n.Stack, x)
	if err != nil {
		return nil, 0, err
	}
	if out == nil || len(out.Shape) < 2 {
		return nil, 0, fmt.Errorf("sprint: bad logits shape %v", outShape(out))
	}
	batch := out.Shape[0]
	classes := out.Shape[1]
	preds = make([]int, batch)
	sumSoft := 0.0
	probs := make([]float32, classes)
	for b := 0; b < batch; b++ {
		off := b * classes
		best := 0
		bv := out.Data[off]
		for c := 1; c < classes; c++ {
			v := out.Data[off+c]
			if v > bv {
				bv, best = v, c
			}
		}
		preds[b] = best
		if target != nil && len(target.Data) >= off+classes {
			lab := 0
			for c := 1; c < classes; c++ {
				if target.Data[off+c] > target.Data[off+lab] {
					lab = c
				}
			}
			softmaxInto(out.Data[off:off+classes], probs)
			sumSoft += metrics.SoftAccProb(probs[lab], 1.0)
		}
	}
	if batch > 0 {
		softAcc = sumSoft / float64(batch)
	}
	return preds, softAcc, nil
}

func (n *StackNet) WeightBytes() int64 {
	if n == nil {
		return 0
	}
	return opBytes(n.Stack)
}

func outShape(t *core.Tensor[float32]) []int {
	if t == nil {
		return nil
	}
	return t.Shape
}

func softmaxInto(logits, out []float32) {
	n := len(logits)
	if n == 0 || len(out) < n {
		return
	}
	max := logits[0]
	for i := 1; i < n; i++ {
		if logits[i] > max {
			max = logits[i]
		}
	}
	var sum float64
	for i := 0; i < n; i++ {
		out[i] = float32(math.Exp(float64(logits[i] - max)))
		sum += float64(out[i])
	}
	if sum <= 0 {
		for i := 0; i < n; i++ {
			out[i] = 1 / float32(n)
		}
		return
	}
	inv := float32(1 / sum)
	for i := 0; i < n; i++ {
		out[i] *= inv
	}
}

func opBytes(op any) int64 {
	if op == nil {
		return 0
	}
	switch v := op.(type) {
	case *parallel.Stack:
		var n int64
		for _, ch := range v.Children {
			n += opBytes(ch)
		}
		return n
	case *dense.Layer:
		return storeBytes(v.Weights)
	case *cnn1.Layer:
		if v.Proj != nil {
			return storeBytes(v.Proj.Weights)
		}
	case *cnn2.Layer:
		if v.Proj != nil {
			return storeBytes(v.Proj.Weights)
		}
	case *cnn3.Layer:
		if v.Proj != nil {
			return storeBytes(v.Proj.Weights)
		}
	case *rnn.Layer:
		return opBytes(v.IH) + opBytes(v.HH)
	case *lstm.Layer:
		var n int64
		if v.I != nil {
			n += opBytes(v.I.IH) + opBytes(v.I.HH)
		}
		if v.F != nil {
			n += opBytes(v.F.IH) + opBytes(v.F.HH)
		}
		if v.G != nil {
			n += opBytes(v.G.IH) + opBytes(v.G.HH)
		}
		if v.O != nil {
			n += opBytes(v.O.IH) + opBytes(v.O.HH)
		}
		return n
	case *mha.Layer:
		return opBytes(v.Q) + opBytes(v.K) + opBytes(v.V) + opBytes(v.O)
	case *mamba.Layer:
		return opBytes(v.InProj) + opBytes(v.OutProj)
	case *swiglu.Layer:
		return opBytes(v.Gate) + opBytes(v.Up) + opBytes(v.Down)
	case *embedding.Layer:
		return storeBytes(v.Weights)
	case *kmeans.Layer:
		return opBytes(v.Centers)
	case *metacognition.Layer:
		return opBytes(v.Observed)
	case *sequential.Layer:
		var n int64
		for _, ch := range v.ChildOps() {
			n += opBytes(ch)
		}
		return n
	case *residual.Layer:
		var n int64
		for _, ch := range v.ChildOps() {
			n += opBytes(ch)
		}
		return n
	case *parallel.Layer:
		var n int64
		for _, ch := range v.Branches {
			n += opBytes(ch)
		}
		return n
	}
	return 0
}

func storeBytes(s *weights.Store) int64 {
	if s == nil {
		return 0
	}
	n := int64(len(s.Bias) * 8)
	if s.Packed != nil {
		n += int64(len(s.Packed.Raw))
		n += int64(len(s.Packed.Scales) * 4)
		n += int64(len(s.Packed.Mins) * 4)
		n += int64(len(s.Packed.Meta))
		return n
	}
	if len(s.Native) > 0 {
		return n + int64(len(s.Native))
	}
	bits := s.DType.Bits()
	if bits <= 0 {
		bits = 32
	}
	return n + int64((s.Rows*s.Cols*bits+7)/8)
}
