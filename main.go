package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/openfluke/quick_sprint/sprint"
	"github.com/openfluke/tide/ocean"
)

func main() {
	oceanAddr := flag.String("ocean", "0.0.0.0:8090", "ocean (master) dashboard address")
	basePort := flag.Int("base-port", 8101, "first layer tide port (dense=base, cnn1=base+1, …)")
	bind := flag.String("bind", "127.0.0.1", "host each layer tide binds")
	mode := flag.String("mode", "sprint", "sprint | smoke | screen | full")
	layers := flag.String("layers", "", "comma list (default: all 22)")
	layer := flag.String("layer", "", "run a single layer tide (no ocean)")
	parallelN := flag.Int("parallel", 4, "how many layer tides train at once")
	trainN := flag.Int("train-n", sprint.TrainN, "synthetic train examples per cell")
	cellMS := flag.Int("cell-ms", sprint.CellMS, "min wall-clock ms per cell (0 = one epoch then stop; 2000 = test48 perm race)")
	autostart := flag.Bool("autostart", true, "signal Start on every layer tide after dashboards are up")
	oceanOnly := flag.Bool("ocean-only", false, "only serve ocean; poll -peers (any existing tides)")
	peers := flag.String("peers", "", "ocean-only: comma list of tide origins")
	fresh := flag.Bool("fresh", false, "ignore per-layer checkpoints")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if *layer != "" {
		opt := sprint.DefaultOptions(*layer)
		opt.Mode = *mode
		opt.TrainN = *trainN
		opt.CellMS = *cellMS
		opt.Autostart = *autostart
		opt.Fresh = *fresh
		if err := sprint.RunLayer(ctx, opt); err != nil && ctx.Err() == nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}

	names := sprint.Names()
	if *layers != "" {
		names = splitCSV(*layers)
	}

	var oceanPeers []ocean.Peer
	if *oceanOnly {
		for i, u := range splitCSV(*peers) {
			oceanPeers = append(oceanPeers, ocean.Peer{Name: fmt.Sprintf("tide-%d", i+1), URL: strings.TrimRight(u, "/")})
		}
		if len(oceanPeers) == 0 {
			fmt.Fprintln(os.Stderr, "ocean-only needs -peers http://127.0.0.1:8080,...")
			os.Exit(2)
		}
	} else {
		n := *parallelN
		if n < 1 {
			n = 1
		}
		sem := make(chan struct{}, n)
		var wg sync.WaitGroup
		for i, name := range names {
			if _, err := sprint.Lookup(name); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(2)
			}
			port := *basePort + i
			addr := net.JoinHostPort(*bind, fmt.Sprintf("%d", port))
			url := "http://" + net.JoinHostPort(loopback(*bind), fmt.Sprintf("%d", port))
			oceanPeers = append(oceanPeers, ocean.Peer{Name: name, URL: url})
			opt := sprint.DefaultOptions(name)
			opt.Addr = addr
			opt.Mode = *mode
			opt.TrainN = *trainN
			opt.CellMS = *cellMS
			opt.Fresh = *fresh
			opt.Autostart = false
			opt.WaitStart = true
			opt.Limit = sem
			wg.Add(1)
			go func(opt sprint.Options) {
				defer wg.Done()
				if err := sprint.RunLayer(ctx, opt); err != nil && ctx.Err() == nil {
					fmt.Fprintf(os.Stderr, "%s: %v\n", opt.Layer, err)
				}
			}(opt)
		}
		_ = wg
	}

	srv := &ocean.Server{
		Addr:   *oceanAddr,
		Title:  "quick_sprint",
		Peers:  oceanPeers,
		OutDir: "results",
	}
	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Println(" quick_sprint — one tide per Welvet layer, ocean consolidates")
	fmt.Printf(" ocean:  %s\n", httpURL(*oceanAddr))
	fmt.Printf(" layers: %d  parallel=%d  mode=%s  train-n=%d  cell-ms=%d\n", len(oceanPeers), *parallelN, *mode, *trainN, *cellMS)
	fmt.Println(" each tide: per-layer toy (XOR / sine / delay / assoc) · Lucy min wall per cell")
	fmt.Println("════════════════════════════════════════════════════════════")
	for _, p := range oceanPeers {
		fmt.Printf("  %-14s %s\n", p.Name, p.URL)
	}
	go func() {
		if err := srv.ListenAndServe(); err != nil {
			fmt.Fprintln(os.Stderr, "ocean:", err)
		}
	}()
	time.Sleep(350 * time.Millisecond)
	fmt.Printf("\nOpen ocean: %s\n", httpURL(*oceanAddr))
	if !*oceanOnly && *autostart {
		fmt.Println("Autostart — signaling every layer /api/start (training queued by -parallel).")
		go startAll(oceanPeers)
	} else {
		fmt.Println("Press Start all on ocean, or Start on a layer tide.")
	}
	<-ctx.Done()
	fmt.Println("\nstopped — per-layer checkpoints under results/<layer>/checkpoint")
}

func startAll(peers []ocean.Peer) {
	time.Sleep(400 * time.Millisecond)
	var wg sync.WaitGroup
	for _, p := range peers {
		wg.Add(1)
		go func(u string) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
			defer cancel()
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, u+"/api/start", nil)
			if err != nil {
				return
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				return
			}
			_ = resp.Body.Close()
		}(p.URL)
	}
	wg.Wait()
}

func splitCSV(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func loopback(bind string) string {
	if bind == "" || bind == "0.0.0.0" || bind == "::" {
		return "127.0.0.1"
	}
	return bind
}

func httpURL(addr string) string {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		if strings.HasPrefix(addr, ":") {
			return "http://127.0.0.1" + addr
		}
		return "http://" + addr
	}
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}
	return "http://" + net.JoinHostPort(host, port)
}
