<!-- Local transcription: operational source-workspace locators were omitted. -->
<!-- Task family: gdn_prefill; original English title: Optimize the Gated Delta Net prefill kernel on the official dev suite (CuTe-DSL) -->

# Optimize the Gated Delta Net prefill kernel on the official dev suite (CuTe-DSL)

## Objective

Implement a Gated Delta Net (chunked gated delta rule) prefill kernel in
CuTe-DSL and maximize BOTH the mean speedup over this task's 34-workload dev
suite and the large geomean, with every workload passing correctness. This
suite is the development proxy for the official MLSys26 contest suite: it
keeps all 18 official large workloads and replaces the 82 official small
workloads with 16 latency-band representatives so a full run stays cheap.
FINAL ACCEPTANCE happens on the full official 100-workload suite
(`gdn_prefill_official`) — optimize transferable properties, not the exact
probe shapes.

Measure the full workload set from inside `[original task locator omitted]`:

    [Original benchmark command omitted.]

## Scoring and submission

Every run prints, and the judge reports, the official-style score over the
dev suite — the arithmetic mean of per-workload speedups vs the flashinfer
baseline, zeroed by any failing workload — plus the all/large/small geomean
diagnostics. A full run additionally prints the weighted ESTIMATED official
mean (each row carries `mean_weight` = its latency band's population divided
by the probes drawn from it; weights sum to 100), which is the number final
acceptance on the full official suite will approximate.

After every change that measurably improves performance, from inside `[original task locator omitted]`
run:

    [Original benchmark command omitted.]

to commit the progress and submit the kernel for authoritative out-of-docker
scoring (it submits only when `[original task locator omitted]` changed and a local smoke check
passes).

## Layout

- `[original task locator omitted]` is a fresh git repository generated from KDA's task recipe. Do all work inside it and
  read its `AGENTS.md` (environment + benchmark policy) before starting.
- Your kernel lives at `[original task locator omitted]` and must expose
  `run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)` — varlen
  bf16 q/k `[T,4,128]`, v `[T,8,128]`, fp32 initial state `[N,8,128,128]`
  (k-last layout), per-head decay parameters, `cu_seqlens[N+1]`. It may
  import sibling files inside `[original task locator omitted]`.
- `[original task locator omitted]` shows the exact interface by
  wrapping the flashinfer baseline (SM100 CuTe-DSL `chunk_gated_delta_rule`).
  You may read it to learn the interface and semantics, but your kernel must
  not call it, nor flashinfer, nor any prebuilt kernel.
- The op: Qwen3-Next linear-attention prefill — chunked gated delta rule with
  grouped value heads (4 q/k heads, 8 v heads, head size 128), varlen
  sequences, per-sequence initial state.
- This container has no GPU of its own. Prefix every GPU command with
  `gpu-run --` (see the AGENTS.md in this directory). `nvcc`/CuTe-DSL
  compilation runs locally without a GPU.

## Rules (binding)

- The kernel must be as self-contained as possible. You may use cute/cutlass
  templates, but all core logic must be written by you inside this repository.
- You must not call pre-existing kernels — in particular no precompiled
  binary artifacts (cubins, prebuilt libraries, downloaded kernels).
- The kernel must be developed entirely in CuTe; other kernel languages such
  as Triton are not allowed.
- NO overfitting to the dev probes: final acceptance re-scores the kernel on
  the full official 100-workload suite, which contains 66 small shapes this
  suite never shows you. Dispatch and tuning may key only on simple size
  classes or thresholds (e.g. total_seq_len buckets, num_seqs ranges) —
  never on the exact (total_seq_len, num_seqs) values of the dev probes. A
  lookup table keyed to the dev shapes is treated as cheating.
- The PyTorch reference is a numerical reference only; speedups are always
  computed against the flashinfer baseline.
- Record every performance-related commit in `benchmark.csv` (produced only
  by `[original task locator omitted]`). Commit early and often so the work history
  stays traceable and analyzable.
- Keep NCU profiling records for each major optimization direction under
  `[original task locator omitted]` and commit them.
- Actively evaluate and use as many relevant Blackwell and CUDA 13.2 features as
  possible, including TMA, TMEM, `tcgen05`, warp specialization, persistent
  scheduling, wide vectorized memory operations, and coalesced memory access
  when they fit the kernel.
- Use KernelWiki for research on Blackwell, CUDA 13.2, linear attention,
  gated delta rule, chunked scan, BF16 attention, TMA, TMEM, and `tcgen05`.
- Use ncu-report-skill when profiling or interpreting Nsight Compute reports.
- Proactively use both Nsight Compute and CuTe-DSL IKET, in separate profiling
  runs, to find concrete bottlenecks and drive further kernel improvements.
- Correctness on every workload is the only hard requirement — one failing
  workload zeroes the official-style score.

## Workloads

34 varlen workloads with real captured `cu_seqlens` and values synthesized
from a seeded RNG, classified by `size_class`:

- **18 large** (`total_seq_len >= 4096`, weight 1.0 each): T 4124..8192,
  N 2..57 — the full official large set, unchanged; the compute/bandwidth
  regime.
- **16 small** (`total_seq_len < 4096`): latency-band representatives of the
  official 82 small workloads, chosen from the frozen per-workload baseline
  latencies. 9 probes sit on the baseline's ~0.13 ms launch-overhead floor
  (T 6..239, N 1..3; in the official suite that floor band is 50 of 100
  workloads), and 7 cover the 0.14-0.45 ms bands up to T=3999 / N=13. Every
  microsecond of fixed per-call overhead (extra launches, host-side syncs,
  allocations, descriptor setup) multiplies across the floor band.

Correctness is checked against the embedded flashinfer baseline,
atol=rtol=1e-2, all elements must match.

Baseline latencies are frozen (pre-measured once per GPU model with the same
CUPTI protocol, shipped in `dataset/flashinfer_trace/frozen_baselines/gdn/`),
so a benchmark run times only YOUR kernel; the baseline still runs live as
the correctness oracle. A full 34-workload run takes roughly 10 minutes and
`--fast` about a minute. The harness runs all shapes in one process: compile
your kernel shape-generically (dynamic T and N); per-shape recompilation will
dominate your runtime.

Development loop: iterate quickly with the fast subset —

    [Original benchmark command omitted.]

which runs a size-balanced dev subset — 2 small + 2 large workloads. Run the
full set before committing a new best.

Whenever you make real progress: commit your kernel, run the FULL recorded
benchmark (first command in Objective; the first full run may take longer
while flashinfer JIT-compiles the correctness baseline — keep the
`--timeout 3600`), and commit `benchmark.csv` + `[original task locator omitted]` in a
follow-up commit. Fast-mode rows are advisory only; only full runs are valid
measurements, and they require a clean committed tree.

## Starting point and DUAL objective — read carefully

`[original task locator omitted]` is pre-seeded (and committed) with the archived GDN prefill
champion (CuTe-DSL, `kernel.py` plus `gdn_*.py` siblings). It was bred under
the old internal metric — large geomean primary — so it is very strong on
the 18 large workloads (about 6.1x large geomean) and gate-clean, but it was
never pressured on the small/tiny regime that dominates the contest mean
(its estimated official mean is about 8.7x; with the baseline's ~0.13 ms
launch floor, well-built tiny-T paths reach 15-25x per workload). This run
optimizes BOTH regimes as CO-PRIMARY objectives, read from every judge
feedback line:

- MEAN: the mean-speedup headline (the "official score" line) over this
  task's 34-workload dev suite — zeroed by any failing workload;
- LARGE: the large geomean over the 18 large workloads (the production
  compute-bound regime the seed is already good at).

Rules of progress (binding for every turn, both agents):

1. First turn only: run the FULL benchmark on the untouched seed, commit
   `[original task locator omitted]`, and `commit_and_submit` it so the seed's authoritative
   MEAN and LARGE values are on record. Those two values minus 1% are HARD
   FLOORS for the rest of the run — no later submission may sit below either
   floor.
2. A submission is a new best ONLY if it improves MEAN or LARGE by at least
   1% over the current best while the other regresses by no more than 1%
   (and both stay above their floors). Correctness on every workload remains
   mandatory at all times — one failure zeroes the score.
3. Maintain `[original task locator omitted]` (committed): one line per accepted
   submission — submission id, MEAN, LARGE, small geomean, and the bench's
   estimated official mean — plus the current best pair and the two floors.
   Read it at the start of every turn; it is the shared state between the
   alternating agents.
4. NO overfitting to the dev probes: FINAL ACCEPTANCE re-scores the kernel
   on the full official 100-workload suite, which contains 66 small shapes
   this dev suite never shows you. Dispatch and tuning may key only on
   simple size classes or thresholds (total_seq_len buckets, num_seqs
   ranges) — never on the exact (total_seq_len, num_seqs) values of the dev
   probes. A lookup keyed to the dev shapes will collapse at final
   acceptance and is treated as cheating.

Where the headroom is: 9 of the 16 small probes sit on the flashinfer
baseline's ~0.13 ms fixed launch floor (T 6..239) — in the real official
suite that floor band is 50 of 100 workloads — so every microsecond of fixed
per-call overhead (extra kernel launches, host-side syncs, allocations,
descriptor setup, per-shape recompilation) is multiplied across half the
final score. Shape-dispatched implementations are explicitly in scope: keep
the champion's large-T path, add a specialized tiny-T fast path (ideally a
single fused launch with precomputed parameters), and route on size classes
inside `run()`. Every path must be correct on every workload. The seed's
large-T design notes are in its kernel docstrings; read them before editing.

All original rules above still apply: CuTe-DSL only, correctness on every
workload, benchmark.csv + [original workspace locator omitted] record keeping, `--fast` (2 small + 2
large) for quick iteration, and the FULL set before every submission.
