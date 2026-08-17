package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"net/http"
	"net/url"
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
	sprint.LoadDotEnv(".env")
	oceanAddr := flag.String("ocean", sprint.EnvOr("TIDE_OCEAN_BIND", "0.0.0.0:8090"), "ocean (master) dashboard address")
	basePort := flag.Int("base-port", 8101, "first layer tide port (dense=base, cnn1=base+1, …)")
	bind := flag.String("bind", sprint.EnvOr("TIDE_BIND", "0.0.0.0"), "host each layer tide binds")
	mode := flag.String("mode", sprint.EnvOr("TIDE_MATRIX", "sprint"), "sprint | smoke | screen | full")
	layers := flag.String("layers", sprint.EnvOr("TIDE_LAYERS", ""), "comma list (default: all 22)")
	layer := flag.String("layer", sprint.EnvOr("TIDE_LAYER", ""), "run a single layer tide (worker; no local ocean)")
	parallelN := flag.Int("parallel", sprint.EnvInt("TIDE_PARALLEL", 4), "how many layer tides train at once")
	trainN := flag.Int("train-n", sprint.TrainN, "synthetic train examples per cell")
	cellMS := flag.Int("cell-ms", sprint.CellMS, "min wall-clock ms per cell (0 = one epoch then stop)")
	modes := flag.String("modes", sprint.EnvOr("TIDE_MODES", ""), "csv train-mode filter (sgd,step_sgd,Sparse,…) empty = all")
	oceanURL := flag.String("ocean-url", sprint.EnvOr("TIDE_OCEAN", ""), "remote ocean master; this tide POSTs /api/register")
	advertise := flag.String("advertise", sprint.EnvOr("TIDE_ADVERTISE", ""), "public origin for ocean to poll (empty = ocean uses this Pi's source IP + port)")
	peerName := flag.String("name", sprint.EnvOr("TIDE_NAME", ""), "ocean peer id (default: layer or layer-modes)")
	addr := flag.String("addr", sprint.EnvOr("TIDE_ADDR", ""), "worker listen address (default 0.0.0.0:8080)")
	oceanOnly := flag.Bool("ocean-only", false, "only serve ocean; poll -peers and/or wait for worker /api/register")
	autostart := flag.Bool("autostart", true, "start training without waiting for ocean Start all")
	waitStart := flag.Bool("wait-start", false, "wait for /api/start (implied when -ocean-url is set)")
	peers := flag.String("peers", sprint.EnvOr("TIDE_PEERS", ""), "ocean-only: comma list of tide origins (optional if workers register)")
	fresh := flag.Bool("fresh", false, "ignore per-layer checkpoints")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if *layer != "" {
		opt := sprint.DefaultOptions(*layer)
		opt.Mode = *mode
		opt.TrainN = *trainN
		opt.CellMS = *cellMS
		opt.Modes = *modes
		opt.OceanURL = *oceanURL
		opt.Advertise = *advertise
		opt.Name = *peerName
		opt.Fresh = *fresh
		if *addr != "" {
			opt.Addr = *addr
		}
		if *oceanURL != "" || *waitStart {
			opt.WaitStart = true
			opt.Autostart = *autostart && *oceanURL == ""
		} else {
			opt.Autostart = *autostart
		}
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
	n := *parallelN
	if n < 1 {
		n = 1
	}

	// Remote ocean + a layer list: one process, slot-queues the collection, no local ocean.
	if !*oceanOnly && strings.TrimSpace(*oceanURL) != "" {
		fmt.Println("════════════════════════════════════════════════════════════")
		fmt.Println(" quick_sprint — farm worker (no local ocean)")
		fmt.Printf(" ocean:  %s\n", strings.TrimRight(*oceanURL, "/"))
		fmt.Printf(" layers: %s\n", strings.Join(names, ", "))
		fmt.Printf(" parallel=%d  matrix=%s  modes=%s  train-n=%d  cell-ms=%d\n",
			n, *mode, nzStar(*modes), *trainN, *cellMS)
		fmt.Println(" one Ctrl+C stops every layer on this box")
		fmt.Println("════════════════════════════════════════════════════════════")
		spawnLayers(ctx, names, layerSpawn{
			bind:      *bind,
			basePort:  *basePort,
			mode:      *mode,
			trainN:    *trainN,
			cellMS:    *cellMS,
			modes:     *modes,
			fresh:     *fresh,
			parallel:  n,
			oceanURL:  *oceanURL,
			advertise: *advertise,
			waitStart: true,
			autostart: *autostart && *oceanURL == "",
		})
		<-ctx.Done()
		fmt.Println("\nstopped — per-layer checkpoints under results/<layer>/checkpoint")
		return
	}

	var oceanPeers []ocean.Peer
	if *oceanOnly {
		for i, u := range splitCSV(*peers) {
			oceanPeers = append(oceanPeers, ocean.Peer{Name: fmt.Sprintf("tide-%d", i+1), URL: strings.TrimRight(u, "/")})
		}
	} else {
		oceanPeers = spawnLayers(ctx, names, layerSpawn{
			bind:      *bind,
			basePort:  *basePort,
			mode:      *mode,
			trainN:    *trainN,
			cellMS:    *cellMS,
			modes:     *modes,
			fresh:     *fresh,
			parallel:  n,
			waitStart: true,
			autostart: false,
		})
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
	fmt.Printf(" layers: %d  parallel=%d  matrix=%s  modes=%s  train-n=%d  cell-ms=%d\n",
		len(oceanPeers), *parallelN, *mode, nzStar(*modes), *trainN, *cellMS)
	fmt.Println(" workers POST /api/register · ocean Start all kicks them")
	fmt.Println("════════════════════════════════════════════════════════════")
	if len(oceanPeers) == 0 {
		fmt.Println("  (no static peers — waiting for worker /api/register)")
	}
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
	} else if *oceanOnly {
		fmt.Println("Master — workers register, then press Start all.")
	} else {
		fmt.Println("Press Start all on ocean, or Start on a layer tide.")
	}
	<-ctx.Done()
	fmt.Println("\nstopped — per-layer checkpoints under results/<layer>/checkpoint")
}

type layerSpawn struct {
	bind      string
	basePort  int
	mode      string
	trainN    int
	cellMS    int
	modes     string
	fresh     bool
	parallel  int
	oceanURL  string
	advertise string
	waitStart bool
	autostart bool
}

func spawnLayers(ctx context.Context, names []string, spec layerSpawn) []ocean.Peer {
	if spec.parallel < 1 {
		spec.parallel = 1
	}
	sem := make(chan struct{}, spec.parallel)
	var peers []ocean.Peer
	for i, name := range names {
		if _, err := sprint.Lookup(name); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		port := spec.basePort + i
		addr := net.JoinHostPort(spec.bind, fmt.Sprintf("%d", port))
		local := "http://" + net.JoinHostPort(loopback(spec.bind), fmt.Sprintf("%d", port))
		peers = append(peers, ocean.Peer{Name: name, URL: local})
		opt := sprint.DefaultOptions(name)
		opt.Addr = addr
		opt.Mode = spec.mode
		opt.TrainN = spec.trainN
		opt.CellMS = spec.cellMS
		opt.Modes = spec.modes
		opt.Fresh = spec.fresh
		opt.OceanURL = spec.oceanURL
		opt.Advertise = advertiseFor(spec.advertise, addr)
		opt.Autostart = spec.autostart
		opt.WaitStart = spec.waitStart
		opt.Limit = sem
		go func(opt sprint.Options) {
			if err := sprint.RunLayer(ctx, opt); err != nil && ctx.Err() == nil {
				fmt.Fprintf(os.Stderr, "%s: %v\n", opt.Layer, err)
			}
		}(opt)
	}
	return peers
}

func advertiseFor(base, listen string) string {
	base = strings.TrimSpace(base)
	if base == "" {
		return ""
	}
	u, err := url.Parse(base)
	if err != nil || u.Hostname() == "" {
		return ""
	}
	_, port, err := net.SplitHostPort(listen)
	if err != nil || port == "" {
		return strings.TrimRight(base, "/")
	}
	scheme := u.Scheme
	if scheme == "" {
		scheme = "http"
	}
	return scheme + "://" + net.JoinHostPort(u.Hostname(), port)
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

func nzStar(s string) string {
	if strings.TrimSpace(s) == "" {
		return "all"
	}
	return s
}
