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

func TestBuildCameralArches(t *testing.T) {
	for _, spec := range []string{"dense", "cnn2", "softmax", "kmeans", "parallel", "layernorm"} {
		for _, arch := range permute.AllArches() {
			c := testCell()
			c.Arch = arch
			c.Cams = permute.CamsOf(arch)
			c.ID = c.String()
			net, err := BuildNet(spec, c)
			if err != nil {
				t.Fatalf("%s %s: %v", spec, arch, err)
			}
			ds := newSynth(mustSpec(spec), Batch, Batch, Batch, 2)
			s := ds.NextServe("A")
			if _, err := net.TrainStep(s.X, s.Target, 0.05, permute.ModeSGD); err != nil {
				t.Fatalf("%s %s train: %v", spec, arch, err)
			}
			if _, _, err := net.ServeEval(s.X, s.Target); err != nil {
				t.Fatalf("%s %s serve: %v", spec, arch, err)
			}
		}
	}
}

func mustSpec(name string) Spec {
	s, err := Lookup(name)
	if err != nil {
		panic(err)
	}
	return s
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

func TestModeFilterShrinksMatrix(t *testing.T) {
	opt := DefaultOptions("dense")
	opt.Modes = "sgd,step_sgd"
	if opt.peerName() != "dense-sgd_step_sgd" {
		t.Fatalf("peer %q", opt.peerName())
	}
	only, err := permute.ParseModes(opt.Modes)
	if err != nil {
		t.Fatal(err)
	}
	full, _ := cellsFor("sprint", nil)
	part, cfg := cellsFor("sprint", only)
	if len(cfg.Modes) != 2 {
		t.Fatalf("modes %v", cfg.Modes)
	}
	if len(part) == 0 || len(part) >= len(full) {
		t.Fatalf("part %d full %d", len(part), len(full))
	}
}
