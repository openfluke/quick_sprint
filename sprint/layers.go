package sprint

import (
	"fmt"
	"hash/fnv"
	"math"
	"math/rand/v2"

	"github.com/openfluke/tide/permute"
	"github.com/openfluke/tide/runner"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

func BuildNet(layer string, cell permute.Cell) (runner.Net, error) {
	spec, err := Lookup(layer)
	if err != nil {
		return nil, err
	}
	if layer == "gdn" && !gdn.PermutationOK(cell.DType, cell.Format, core.BackendSIMD) {
		return nil, fmt.Errorf("gdn: permutation not supported (%s / %s) — float32 only", cell.DType, cell.Format)
	}
	h := fnv.New64a()
	_, _ = h.Write([]byte(cell.ID + layer))
	rng := rand.New(rand.NewPCG(h.Sum64(), 0x51A7E))
	stack, err := buildStack(spec, rng)
	if err != nil {
		return nil, err
	}
	stack.Exec.Backend = core.BackendSIMD
	stack.Exec.MultiCore = true
	stack.SyncChildExec()
	if cell.Format != quant.FormatNone {
		if err := stack.Pack(cell.Format); err != nil {
			return nil, err
		}
	} else if cell.DType != core.DTypeFloat32 {
		if err := stack.SetDType(cell.DType); err != nil {
			return nil, err
		}
	}
	return &StackNet{Stack: stack}, nil
}

func randN(n int, rng *rand.Rand) []float32 {
	w := make([]float32, n)
	scale := float32(1 / math.Sqrt(float64(n)))
	if scale > 0.1 {
		scale = 0.1
	}
	for i := range w {
		w[i] = (rng.Float32()*2 - 1) * scale
	}
	return w
}

func denseHead(in int, rng *rand.Rand) (*dense.Layer, error) {
	return dense.NewConfigured(in, Classes, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, randN(in*Classes, rng))
}

func stackOf(kids ...any) (*parallel.Stack, error) {
	return parallel.NewStack(kids...)
}

func withViewHead(subject any, flat int, rng *rand.Rand) (*parallel.Stack, error) {
	v, err := parallel.NewView(Batch, flat)
	if err != nil {
		return nil, err
	}
	h, err := denseHead(flat, rng)
	if err != nil {
		return nil, err
	}
	return stackOf(subject, v, h)
}

func withHead(subject any, in int, rng *rand.Rand) (*parallel.Stack, error) {
	h, err := denseHead(in, rng)
	if err != nil {
		return nil, err
	}
	return stackOf(subject, h)
}

func buildStack(spec Spec, rng *rand.Rand) (*parallel.Stack, error) {
	switch spec.Name {
	case "dense":
		d, err := dense.NewConfigured(8, Classes, core.ActivationTanh, core.DTypeFloat32, quant.FormatNone, randN(8*Classes, rng))
		if err != nil {
			return nil, err
		}
		return stackOf(d)
	case "cnn1":
		cfg := cnn1.Config{InChannels: 1, Filters: 4, SeqLen: 16, Kernel: 3, Stride: 1, Activation: core.ActivationTanh}
		l, err := cnn1.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, randN(cfg.Filters*cfg.PatchDim(), rng))
		if err != nil {
			return nil, err
		}
		return withViewHead(l, cfg.Filters*cfg.OutLen(), rng)
	case "cnn2":
		cfg := cnn2.Config{InChannels: 1, Filters: 2, Height: 6, Width: 6, Kernel: 3, Padding: 0, Activation: core.ActivationTanh}
		l, err := cnn2.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, randN(cfg.Filters*cfg.PatchDim(), rng))
		if err != nil {
			return nil, err
		}
		return withViewHead(l, cfg.Filters*cfg.OutH()*cfg.OutW(), rng)
	case "cnn3":
		cfg := cnn3.Config{InChannels: 1, Filters: 2, Depth: 4, Height: 4, Width: 4, Kernel: 2, Stride: 2, Activation: core.ActivationTanh}
		l, err := cnn3.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, randN(cfg.Filters*cfg.PatchDim(), rng))
		if err != nil {
			return nil, err
		}
		return withViewHead(l, cfg.Filters*cfg.OutD()*cfg.OutH()*cfg.OutW(), rng)
	case "convt1":
		cfg := convt1.Config{InChannels: 2, Filters: 2, SeqLen: 4, Kernel: 3, Stride: 2, OutputPadding: 1, Activation: core.ActivationTanh}
		l, err := convt1.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, randN(cfg.Filters*cfg.PatchDim(), rng))
		if err != nil {
			return nil, err
		}
		return withViewHead(l, cfg.Filters*cfg.OutLen(), rng)
	case "convt2":
		cfg := convt2.Config{InChannels: 2, Filters: 2, Height: 2, Width: 2, Kernel: 3, Stride: 2, OutputPadding: 1, Activation: core.ActivationTanh}
		l, err := convt2.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, randN(cfg.Filters*cfg.PatchDim(), rng))
		if err != nil {
			return nil, err
		}
		return withViewHead(l, cfg.Filters*cfg.OutH()*cfg.OutW(), rng)
	case "convt3":
		cfg := convt3.Config{InChannels: 2, Filters: 2, Depth: 2, Height: 2, Width: 2, Kernel: 2, Stride: 1, Activation: core.ActivationTanh}
		l, err := convt3.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, randN(cfg.Filters*cfg.PatchDim(), rng))
		if err != nil {
			return nil, err
		}
		return withViewHead(l, cfg.Filters*cfg.OutD()*cfg.OutH()*cfg.OutW(), rng)
	case "rnn":
		cfg := rnn.Config{InputSize: 4, HiddenSize: 8, SeqLen: 8}
		l, err := rnn.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, randN(cfg.WeightCount(), rng))
		if err != nil {
			return nil, err
		}
		return withViewHead(l, cfg.SeqLen*cfg.HiddenSize, rng)
	case "lstm":
		cfg := lstm.Config{InputSize: 4, HiddenSize: 8, SeqLen: 8}
		l, err := lstm.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, randN(cfg.WeightCount(), rng))
		if err != nil {
			return nil, err
		}
		return withViewHead(l, cfg.SeqLen*cfg.HiddenSize, rng)
	case "mha":
		cfg := mha.Config{
			DModel: 8, NumHeads: 2, MaxSeqLen: 8,
			Mask: mha.MaskBidirectional, Pos: mha.PosNone, Mode: mha.ModeSelf,
			Role: mha.RoleEncoderSelf,
		}
		q := randN(8*8, rng)
		l, err := mha.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, q, randN(64, rng), randN(64, rng), randN(64, rng))
		if err != nil {
			return nil, err
		}
		return withViewHead(l, 8*8, rng)
	case "mamba":
		cfg := mamba.Config{DModel: 8, DState: 4, Expand: 2, SeqLen: 8}
		inner := cfg.InnerDim()
		l, err := mamba.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone,
			randN(2*inner*8, rng), randN(8*inner, rng), nil, nil)
		if err != nil {
			return nil, err
		}
		return withViewHead(l, 8*8, rng)
	case "gdn":
		l, err := gdn.New(gdn.Config{
			HiddenSize: 8, NumKeyHeads: 1, NumValueHeads: 1,
			KeyHeadDim: 8, ValueHeadDim: 8, ConvKernel: 4,
		})
		if err != nil {
			return nil, err
		}
		return withViewHead(l, 8*8, rng)
	case "embedding":
		cfg := embedding.Config{VocabSize: 16, EmbeddingDim: 8, SeqLen: 8}
		l, err := embedding.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, randN(cfg.WeightCount(), rng))
		if err != nil {
			return nil, err
		}
		return withViewHead(l, cfg.SeqLen*cfg.EmbeddingDim, rng)
	case "swiglu":
		cfg := swiglu.Config{InputDim: 8, IntermediateDim: 16}
		l, err := swiglu.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone,
			randN(16*8, rng), randN(16*8, rng), randN(8*16, rng))
		if err != nil {
			return nil, err
		}
		return withHead(l, 8, rng)
	case "layernorm":
		stem, err := dense.NewConfigured(8, 8, core.ActivationTanh, core.DTypeFloat32, quant.FormatNone, randN(64, rng))
		if err != nil {
			return nil, err
		}
		gamma := make([]float32, 8)
		for i := range gamma {
			gamma[i] = 1
		}
		ln, err := layernorm.NewConfigured(layernorm.Config{Dim: 8}, core.DTypeFloat32, quant.FormatNone, gamma, nil)
		if err != nil {
			return nil, err
		}
		h, err := denseHead(8, rng)
		if err != nil {
			return nil, err
		}
		return stackOf(stem, ln, h)
	case "rmsnorm":
		stem, err := dense.NewConfigured(8, 8, core.ActivationTanh, core.DTypeFloat32, quant.FormatNone, randN(64, rng))
		if err != nil {
			return nil, err
		}
		gamma := make([]float32, 8)
		for i := range gamma {
			gamma[i] = 1
		}
		rn, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: 8}, core.DTypeFloat32, quant.FormatNone, gamma)
		if err != nil {
			return nil, err
		}
		h, err := denseHead(8, rng)
		if err != nil {
			return nil, err
		}
		return stackOf(stem, rn, h)
	case "softmax":
		d, err := dense.NewConfigured(8, Classes, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, randN(8*Classes, rng))
		if err != nil {
			return nil, err
		}
		sm, err := softmax.New(softmax.Config{Dim: Classes})
		if err != nil {
			return nil, err
		}
		return stackOf(d, sm)
	case "kmeans":
		l, err := kmeans.NewConfigured(kmeans.Config{
			NumClusters: Classes, FeatureDim: 8, OutputMode: kmeans.OutputProbabilities,
		}, core.DTypeFloat32, quant.FormatNone, randN(Classes*8, rng))
		if err != nil {
			return nil, err
		}
		return stackOf(l)
	case "sequential":
		a, err := dense.NewConfigured(8, 8, core.ActivationTanh, core.DTypeFloat32, quant.FormatNone, randN(64, rng))
		if err != nil {
			return nil, err
		}
		b, err := dense.NewConfigured(8, 8, core.ActivationTanh, core.DTypeFloat32, quant.FormatNone, randN(64, rng))
		if err != nil {
			return nil, err
		}
		seq, err := sequential.NewFromOps(sequential.Config{Dim: 8, Depth: 2}, []any{a, b})
		if err != nil {
			return nil, err
		}
		return withHead(seq, 8, rng)
	case "residual":
		f, err := dense.NewConfigured(8, 8, core.ActivationTanh, core.DTypeFloat32, quant.FormatNone, randN(64, rng))
		if err != nil {
			return nil, err
		}
		res, err := residual.NewFromOps(residual.Config{Dim: 8, Depth: 1}, []any{f})
		if err != nil {
			return nil, err
		}
		return withHead(res, 8, rng)
	case "parallel":
		stem, err := dense.NewConfigured(8, 8, core.ActivationTanh, core.DTypeFloat32, quant.FormatNone, randN(64, rng))
		if err != nil {
			return nil, err
		}
		b0, err := dense.NewConfigured(8, 8, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, randN(64, rng))
		if err != nil {
			return nil, err
		}
		b1, err := dense.NewConfigured(8, 8, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, randN(64, rng))
		if err != nil {
			return nil, err
		}
		para, err := parallel.NewFromBranches(parallel.Config{
			Dim: 8, OutFeat: 8, Branches: 2, Combine: parallel.CombineAdd,
		}, []any{b0, b1}, nil)
		if err != nil {
			return nil, err
		}
		h, err := denseHead(8, rng)
		if err != nil {
			return nil, err
		}
		return stackOf(stem, para, h)
	case "metacognition":
		mc, err := metacognition.NewConfigured(metacognition.Config{Dim: 8}, core.DTypeFloat32, quant.FormatNone, randN(64, rng))
		if err != nil {
			return nil, err
		}
		return withHead(mc, 8, rng)
	default:
		return nil, fmt.Errorf("no stack builder for %s", spec.Name)
	}
}
