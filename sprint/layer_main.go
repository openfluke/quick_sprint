package sprint

import (
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"context"
)

// LayerMain is the entrypoint for each per-layer folder (`go run ./dense`).
func LayerMain(layer string) {
	opt := DefaultOptions(layer)
	addr := flag.String("addr", opt.Addr, "dashboard listen address")
	mode := flag.String("mode", opt.Mode, "sprint | smoke | screen | full")
	ckpt := flag.String("ckpt", opt.CkptDir, "checkpoint directory")
	trainN := flag.Int("train-n", opt.TrainN, "synthetic train examples per cell")
	cellMS := flag.Int("cell-ms", opt.CellMS, "min wall-clock ms per cell (0 = one epoch then stop)")
	lr := flag.Float64("lr", opt.LR, "learning rate")
	pulseMS := flag.Int("pulse-ms", opt.PulseMS, "Lucy pulse interval (ms)")
	fresh := flag.Bool("fresh", false, "ignore checkpoint")
	autostart := flag.Bool("autostart", true, "start training without the dashboard button")
	wait := flag.Bool("wait-start", false, "ignore autostart and wait for /api/start (ocean Start all)")
	flag.Parse()
	opt.Addr = *addr
	opt.Mode = *mode
	opt.CkptDir = *ckpt
	opt.TrainN = *trainN
	opt.CellMS = *cellMS
	opt.LR = *lr
	opt.PulseMS = *pulseMS
	opt.Fresh = *fresh
	opt.Autostart = *autostart
	opt.WaitStart = *wait

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := RunLayer(ctx, opt); err != nil && ctx.Err() == nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
