package sprint

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/openfluke/tide/checkpoint"
	"github.com/openfluke/tide/dash"
	"github.com/openfluke/tide/permute"
	"github.com/openfluke/tide/pulse"
	"github.com/openfluke/tide/report"
	"github.com/openfluke/tide/runner"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
)

// Options is one layer tide.
type Options struct {
	Layer     string
	Addr      string
	Mode      string // sprint | smoke | screen | full
	CkptDir   string
	TrainN    int
	Batch     int // permute dashboard batch
	Micro     int
	LR        float64
	PulseMS   int
	CkptSec   int
	Fresh     bool
	Autostart bool
	WaitStart bool // if true, block on dashboard Start even when Autostart
	Limit     chan struct{} // optional worker pool (ocean orchestrator)
}

func DefaultOptions(layer string) Options {
	return Options{
		Layer:     layer,
		Addr:      "0.0.0.0:8080",
		Mode:      "sprint",
		CkptDir:   "results/" + layer + "/checkpoint",
		TrainN:    TrainN,
		Batch:     8,
		Micro:     Batch,
		LR:        0.05,
		PulseMS:   50,
		CkptSec:   30,
		Autostart: true,
	}
}

func cellsFor(mode string) ([]permute.Cell, permute.Config) {
	var pcfg permute.Config
	switch mode {
	case "full":
		pcfg = permute.Full()
		pcfg.Arches = permute.AllArches()
	case "screen":
		pcfg = permute.Screen()
		pcfg.Arches = permute.AllArches()
	case "smoke":
		pcfg = permute.Smoke()
		pcfg.Arches = permute.AllArches()
	default:
		pcfg = permute.Sprint()
	}
	if len(pcfg.Formats) == 0 {
		pcfg.Formats = []quant.Format{quant.FormatNone}
	}
	return permute.Expand(pcfg), pcfg
}

// RunLayer starts this layer's tide dashboard + Lucy runner. Blocks until ctx done
// or the epoch finishes (then keeps serving until ctx cancel).
func RunLayer(ctx context.Context, opt Options) error {
	spec, err := Lookup(opt.Layer)
	if err != nil {
		return err
	}
	if opt.Addr == "" {
		opt.Addr = "0.0.0.0:8080"
	}
	if opt.Mode == "" {
		opt.Mode = "sprint"
	}
	if opt.TrainN < Batch {
		opt.TrainN = TrainN
	}
	if opt.Micro < 1 {
		opt.Micro = Batch
	}
	if opt.PulseMS < 1 {
		opt.PulseMS = 50
	}
	cells, _ := cellsFor(opt.Mode)

	store := checkpoint.New(opt.CkptDir, opt.Mode)
	var resume *checkpoint.Progress
	if !opt.Fresh {
		resume, err = store.Load()
		if err != nil {
			return fmt.Errorf("checkpoint: %w", err)
		}
	}
	epoch, resume, skipTrain := pinOneEpoch(resume, cells)

	tr := pulse.New()
	srv := &dash.Server{
		Tracker:  tr,
		Cells:    cells,
		Addr:     opt.Addr,
		Epoch:    epoch,
		Task:     spec.Name,
		ID:       spec.Name,
		Subtitle: fmt.Sprintf("%s · 1 epoch · %d train · pulse %dms · SIMD %v", spec.Strength, opt.TrainN, opt.PulseMS, simd.Enabled()),
	}
	go func() {
		if err := srv.ListenAndServe(); err != nil {
			fmt.Fprintf(os.Stderr, "%s dash: %v\n", spec.Name, err)
		}
	}()

	cfg := runner.DefaultConfig(cells)
	cfg.BatchSize = opt.Batch
	cfg.Epoch = epoch
	cfg.PulseEvery = time.Duration(opt.PulseMS) * time.Millisecond
	cfg.CheckpointEvery = time.Duration(opt.CkptSec) * time.Second
	if cfg.CheckpointEvery <= 0 {
		cfg.CheckpointEvery = 30 * time.Second
	}
	cfg.LR = opt.LR
	cfg.Store = store
	cfg.Resume = resume
	cfg.Build = func(cell permute.Cell) (runner.Net, error) {
		return BuildNet(spec.Name, cell)
	}

	ds := newSynth(spec, opt.TrainN, ValN, opt.Micro, 0x51524E54)

	fmt.Printf(" tide %s  %s  cells=%d  dash=%s  simd=%v\n",
		spec.Name, spec.Strength, len(cells), dashURLs(opt.Addr), simd.Enabled())

	doneN := len(checkpoint.DoneSet(resume))
	runner.Hydrate(tr, cfg, fmt.Sprintf(
		"paused — epoch %d — %d/%d done — press Start",
		epoch, doneN, len(cells)))

	if skipTrain {
		srv.SignalStart()
		tr.SetMeta(0, 0, len(cells), len(cells),
			"epoch 1 done — one-epoch sprint (slot free for other layers)")
		fmt.Printf(" tide %s already finished epoch 1 — dashboard up, not training again\n", spec.Name)
		writeLayerPDF(srv, opt, spec.Name, epoch)
		<-ctx.Done()
		return ctx.Err()
	}

	if opt.Autostart && !opt.WaitStart {
		srv.SignalStart()
	} else {
		if err := srv.AwaitStart(ctx); err != nil {
			return err
		}
	}
	if opt.Limit != nil {
		tr.SetMeta(0, 0, 0, len(cells),
			fmt.Sprintf("queued — waiting for a worker slot · epoch %d · %d cells", epoch, len(cells)))
		select {
		case opt.Limit <- struct{}{}:
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	runErr := runner.Run(ctx, cfg, ds, tr)
	// Release the worker slot as soon as this epoch is done so the next
	// queued layer can train. Keep the dashboard up until Ctrl+C.
	if opt.Limit != nil {
		<-opt.Limit
	}
	if runErr != nil && ctx.Err() == nil {
		return runErr
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	fmt.Printf(" tide %s epoch %d complete — dashboard still up, worker slot released\n", spec.Name, epoch)
	tr.SetMeta(0, 0, len(cells), len(cells),
		fmt.Sprintf("epoch %d done — one-epoch sprint (slot free for other layers)", epoch))
	writeLayerPDF(srv, opt, spec.Name, epoch)
	<-ctx.Done()
	return ctx.Err()
}

func writeLayerPDF(srv *dash.Server, opt Options, name string, epoch int) {
	pdf, err := report.PDFTide(srv.Report())
	if err != nil {
		return
	}
	dir := filepath.Dir(opt.CkptDir)
	if dir == "" || dir == "." {
		dir = "results/" + name
	}
	_ = os.MkdirAll(dir, 0o755)
	path := filepath.Join(dir, fmt.Sprintf("report-epoch%d.pdf", epoch))
	if werr := os.WriteFile(path, pdf, 0o644); werr == nil {
		fmt.Printf(" tide %s wrote %s\n", name, path)
	}
}

// pinOneEpoch keeps quick_sprint on a single sweep. New cameral/mode IDs still
// run; a finished matrix does not start epoch 2.
func pinOneEpoch(resume *checkpoint.Progress, cells []permute.Cell) (epoch int, out *checkpoint.Progress, done bool) {
	if checkpoint.AllDone(resume, cells) {
		return 1, resume, true
	}
	epoch, out = checkpoint.PrepareEpoch(resume, cells)
	if epoch > 1 {
		epoch = 1
		if out != nil {
			out.Epoch = 1
			if len(out.DoneIDs) == 0 {
				out.DoneIDs = doneIDsFromCompleted(out)
			}
		}
		if checkpoint.AllDone(out, cells) {
			return 1, out, true
		}
	}
	return epoch, out, false
}

func doneIDsFromCompleted(p *checkpoint.Progress) []string {
	if p == nil {
		return nil
	}
	seen := map[string]bool{}
	var ids []string
	for _, r := range p.Completed {
		id := r.Cell.ID
		if id == "" || seen[id] {
			continue
		}
		seen[id] = true
		ids = append(ids, id)
	}
	return ids
}

func dashURLs(addr string) string {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		if strings.HasPrefix(addr, ":") {
			host, port = "", strings.TrimPrefix(addr, ":")
		} else {
			return "http://" + addr
		}
	}
	if port == "" {
		port = "8080"
	}
	all := host == "" || host == "0.0.0.0" || host == "::"
	if !all {
		return "http://" + net.JoinHostPort(host, port)
	}
	return fmt.Sprintf("http://127.0.0.1:%s", port)
}
