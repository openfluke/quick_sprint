# quick_sprint

Tiny **Lucy** serve+train sprints of **every Welvet layer**, each hosted as its
own **tide** dashboard. A master **ocean** dashboard links them and consolidates
the best **train mode** and **dtype** per layer, then holistically.

This is the cheap cousin of [`live_mnist`](../live_mnist): synthetic 4-class
data (256 train examples), `permute.Sprint()` (all dtypes × FormatNone × all
train modes × cnn-arch token). Not the month-long Full matrix.

Tide itself is unchanged for MNIST. `runner.Config.Build` is optional; live_mnist
leaves it nil and still uses `chain.Model`.

## Watch everything

```bash
cd quick_sprint
go run .
# open http://127.0.0.1:8090          ocean (master)
#      http://127.0.0.1:8101          dense tide
#      http://127.0.0.1:8102          cnn1
#      …
```

Ocean polls each tide’s `/api/board` (CORS-open JSON). Press **Start all** or
use `-autostart` (default). `-parallel 4` keeps only four layers training at
once so the box does not melt. When a layer **finishes its epoch**, it frees
that slot (the dashboard stays up so you can still open it). Restart the
process after this change if an older run looks frozen at 4/22 done.

## One layer

```bash
go run ./dense -addr :8101
# or
go run . -layer dense
```

Each layer folder is a thin `main` that calls `sprint.LayerMain("dense")`.

## Link an already-running tide (live_mnist)

```bash
# terminal 1
cd ../live_mnist && go run . -addr :8080 -mode screen -autostart

# terminal 2
cd ../quick_sprint
go run . -ocean-only -peers http://127.0.0.1:8080
```

Ocean can list any mix of live_mnist + layer tides.

## Matrix

| Flag | Cells (approx) | Notes |
|------|----------------|-------|
| `-mode sprint` (default) | ~782 / layer | 34 dtypes × 23 modes × FormatNone |
| `-mode smoke` | small | dashboard check |
| `-mode screen` | Lucy 6 × numeric | closer to live_mnist screen |
| `-mode full` | packed quants too | still ArchCNN only here |

Pulse is **50ms** so a ~200–400ms cell still records Lucy windows, Score,
Availability, SoftAcc, time-to-acc.

## APIs on every tide (including live_mnist)

| Path | What |
|------|------|
| `GET /api/live` | full pulse + winners + `id` / `apis` |
| `GET /api/board` | compact snapshot ocean polls |
| `GET /api/meta` | identity |
| `GET /api/winners` | axis champion tables |
| `POST /api/start` | release the pause gate |
| `GET /api/report` | JSON Lucy snapshot (winners, leaderboard, pulse history) |
| `GET /api/report.pdf` | per-tide PDF of those metrics and graphs |

Ocean `GET /api/report.pdf` pulls every live tide report and stitches one master PDF (also written to `results/ocean-report.pdf`). Each finished epoch also writes `results/<layer>/report-epochN.pdf`.

The progress bar is **this epoch** (`done/plan`, usually 782). A number like `1073/782` was recorded cells across epochs — epoch 1 finished, epoch 2 kept going. The bar is now capped at 100% and the label shows recorded separately.

CORS `*` is on so another page can fetch a tide directly.

## Findings

Ocean’s holistic vote is: **plurality of per-tide best-Score cells**, tie-break
mean Score — once for train mode, once for dtype. The per-layer table is the
individual-layer answer. Checkpoints stay under `results/<layer>/checkpoint`.
