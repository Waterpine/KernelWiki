<!-- Local transcription: operational source-workspace locators were omitted. -->
<!-- Task family: gdn_prefill; original English title: Optimize the Gated Delta Net prefill kernel on NVIDIA B300 (CuTe-DSL) -->

# Optimize the Gated Delta Net prefill kernel on NVIDIA B300 (CuTe-DSL)

## Experiment-specific language constraint (binding)

The complete core GPU implementation MUST use NVIDIA CuTe DSL. Python may be
used only for CuTe compilation, launch, and thin dispatch glue. Custom CUDA C++
and Triton kernels are forbidden. This rule takes precedence over conflicting
language or profiler clauses elsewhere; every correctness, benchmark,
submission, provenance, anti-cheating, and work-tracking rule remains binding.

## Objective

Implement a Gated Delta Net (chunked gated delta rule) prefill kernel in
CuTe-DSL and make it as fast as possible on the FULL benchmark workload set,
with every workload passing correctness. There is no fixed speedup target and
no acceptance threshold: correctness is the only hard requirement, and speed is
optimized continuously with no finish line.

Measure the full workload set from inside `[original task locator omitted]`:

    [Original benchmark command omitted.]

## Scoring and submission

Speed is scored as three geomean speedups vs the flashinfer baseline — all,
large, and small. The **large** geomean is the primary objective; the **small**
geomean is a reported anti-regression guardrail. Correctness is the only
pass/fail; there is no fixed speedup target.

After every change that measurably improves performance, from inside `[original task locator omitted]`
run:

    [Original benchmark command omitted.]

to commit the progress and submit the kernel for authoritative out-of-docker
scoring (it submits only when `[original task locator omitted]` changed and a local smoke check
passes). Read the returned three geomeans and keep optimizing to push them
higher. Never declare the task complete or stop — keep improving for the entire
session.

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
- The PyTorch reference is a numerical reference only; speedups are always
  computed against the flashinfer baseline.
- Record every performance-related commit in `benchmark.csv` (produced only
  by `[original task locator omitted]`). Commit early and often so the work history
  stays traceable and analyzable.
- Keep available NCU profiling records for major optimization directions under
  `[original task locator omitted]` and commit them.
- Actively evaluate and use as many relevant B300 and CUDA 13.2 features as
  possible, including TMA, TMEM, `tcgen05`, warp specialization, persistent
  scheduling, wide vectorized memory operations, and coalesced memory access
  when they fit the kernel.
- Use KernelWiki for research on Blackwell/B300, CUDA 13.2, Triton, sparse
  attention, MLA/DSA, paged KV cache access, BF16 attention, softmax/LSE,
  TMA, TMEM, and `tcgen05`.
- Use ncu-report-skill when profiling or interpreting Nsight Compute reports.
- Proactively use CuTe-DSL IKET and use Nsight Compute when hardware counters
  are available. If the host denies NCU counters, record that once and continue
  benchmarking.
- There is no fixed speedup target; keep optimizing to push the speedup higher.
  Correctness on every workload is the only hard requirement.

## Workloads

30 varlen workloads, shape-stratified by `size_class`: 18 large and 12 small
(total_seq_len 30..8192, num_seqs 1..57 — large: T 4124..8192, N 2..57; small:
T 30..3999, N 1..13), with real captured `cu_seqlens` and values synthesized
from a seeded RNG. Speed is reported as three geomean speedups vs the flashinfer
baseline — all / large / small. Correctness is checked against the embedded
flashinfer baseline, atol=rtol=1e-2, all elements must match.

Development loop: iterate quickly with the fast subset —

    [Original benchmark command omitted.]

which runs a size-balanced dev subset — 2 small + 2 large workloads — so a
quick iteration exercises both the launch-overhead (small) regime and the
compute-bound (large) regime. Run the full set before committing a new best.

Whenever you make real progress: commit your kernel, run the FULL recorded
benchmark (first command in Objective; the first full run may take much
longer while flashinfer JIT-compiles baseline kernels — keep the
`--timeout 3600`), and commit `benchmark.csv` + `[original task locator omitted]` in a
follow-up commit. Fast-mode rows are advisory only; only full runs are valid
measurements, and they require a clean committed tree.
