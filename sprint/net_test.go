package sprint

import (
	"testing"

	"github.com/openfluke/tide/permute"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
)

func testCell() permute.Cell {
	c := permute.Cell{
		DType:   core.DTypeFloat32,
		Format:  quant.FormatNone,
		Mode:    permute.ModeSGD,
		Arch:    permute.ArchCNN,
		Cams:    1,
		Backend: core.BackendSIMD,
		UseSIMD: true,
	}
	c.ID = c.String()
	return c
}

func TestBuildAndStepLayers(t *testing.T) {
	cell := testCell()
	dsCache := map[string]*synthDS{}
	for _, spec := range AllSpecs() {
		spec := spec
		t.Run(spec.Name, func(t *testing.T) {
			net, err := BuildNet(spec.Name, cell)
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			ds := dsCache[spec.Name]
			if ds == nil {
				ds = newSynth(spec, Batch, Batch, Batch, 1)
				dsCache[spec.Name] = ds
			}
			s := ds.NextServe("A")
			if _, err := net.TrainStep(s.X, s.Target, 0.05, permute.ModeSGD); err != nil {
				t.Fatalf("train: %v", err)
			}
			preds, soft, err := net.ServeEval(s.X, s.Target)
			if err != nil {
				t.Fatalf("serve: %v", err)
			}
			if len(preds) != Batch {
				t.Fatalf("preds %d", len(preds))
			}
			if net.WeightBytes() < 0 {
				t.Fatalf("bytes")
			}
			t.Logf("soft=%.2f bytes=%d", soft, net.WeightBytes())
		})
	}
}

func TestGDNNonFloat32Gaps(t *testing.T) {
	c := testCell()
	c.DType = core.DTypeInt8
	c.ID = c.String()
	_, err := BuildNet("gdn", c)
	if err == nil {
		t.Fatal("expected gdn gap on int8")
	}
}
