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
	LoadDotEnv(".env")
	if env := EnvOr("TIDE_LAYER", ""); env != "" {
		layer = env
	}
	opt := DefaultOptions(layer)
	addr := flag.String("addr", EnvOr("TIDE_ADDR", opt.Addr), "dashboard listen address")
	mode := flag.String("mode", EnvOr("TIDE_MATRIX", opt.Mode), "sprint | smoke | screen | full")
	ckpt := flag.String("ckpt", opt.CkptDir, "checkpoint directory")
	trainN := flag.Int("train-n", opt.TrainN, "synthetic train examples per cell")
	cellMS := flag.Int("cell-ms", opt.CellMS, "min wall-clock ms per cell (0 = one epoch then stop)")
	modes := flag.String("modes", EnvOr("TIDE_MODES", ""), "csv train-mode filter")
	oceanURL := flag.String("ocean-url", EnvOr("TIDE_OCEAN", ""), "remote ocean master")
	advertise := flag.String("advertise", EnvOr("TIDE_ADVERTISE", ""), "public origin ocean should poll")
	name := flag.String("name", EnvOr("TIDE_NAME", ""), "ocean peer id")
	lr := flag.Float64("lr", opt.LR, "learning rate")
	pulseMS := flag.Int("pulse-ms", opt.PulseMS, "Lucy pulse interval (ms)")
	fresh := flag.Bool("fresh", false, "ignore checkpoint")
	autostart := flag.Bool("autostart", true, "start training without the dashboard button")
	wait := flag.Bool("wait-start", false, "ignore autostart and wait for /api/start")
	flag.Parse()
	opt.Addr = *addr
	opt.Mode = *mode
	opt.CkptDir = *ckpt
	opt.TrainN = *trainN
	opt.CellMS = *cellMS
	opt.Modes = *modes
	opt.OceanURL = *oceanURL
	opt.Advertise = *advertise
	opt.Name = *name
	opt.LR = *lr
	opt.PulseMS = *pulseMS
	opt.Fresh = *fresh
	if *oceanURL != "" || *wait {
		opt.WaitStart = true
		opt.Autostart = *autostart && *oceanURL == ""
	} else {
		opt.Autostart = *autostart
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := RunLayer(ctx, opt); err != nil && ctx.Err() == nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
