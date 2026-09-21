<!-- Local transcription: operational source-workspace locators were omitted. -->
<!-- Task family: kda_forward; original English title: Optimize the Kimi Delta Attention forward kernel on NVIDIA Blackwell (CuTe-DSL) -->

# Optimize the Kimi Delta Attention forward kernel on NVIDIA Blackwell (CuTe-DSL)

## Objective

Implement a Kimi Delta Attention (KDA, chunked per-channel-gated delta rule)
forward kernel in CuTe-DSL and make it as fast as possible on the FULL
benchmark workload set, with every workload passing correctness.

Measure the full workload set from inside `[original task locator omitted]`:

    [Original benchmark command omitted.]

## Scoring and submission

All 6 workloads are classified "large". Speed is reported as geomean speedup
vs the baseline — all workloads. Correctness is the only pass/fail.

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
  `run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens)`
  returning the bf16 output `[1, T, H, 128]` — packed varlen format (B=1,
  T=total tokens). Inputs: bf16 `q/k/v/g [1, T, H, 128]`, bf16 beta logits
  `beta [1, T, H]`, fp32 `A_log [H]`, fp32 flattened `dt_bias [H*128]`, python
  float `scale` (1/sqrt(128)), fp32 `initial_state [num_seqs, H, 128, 128]`,
  and `cu_seqlens` (int64 tensor `[num_seqs+1]` of cumulative sequence
  start/end offsets, or `None` for a single sequence). It may import sibling
  files inside `[original task locator omitted]`.
- `[original task locator omitted]` shows the exact interface by
  wrapping the baseline solution. You may read it (and
  `[original task locator omitted]`) to learn the interface and
  semantics, but your kernel must not call it, nor any prebuilt kernel.
- The op: one tensor-parallel rank of Kimi-K3 linear-attention training
  forward — chunked delta rule with per-channel (fine-grained) gating: q/k are
  L2-normalized in-kernel, the per-channel log-decay gate is computed
  in-kernel with K3's bounded form
  `-5.0 * sigmoid(exp(A_log) * (g + dt_bias))`, beta is `sigmoid(beta)`,
  state decays channel-wise along K (V-first recurrent-state layout). No
  final-state output; each sequence starts from its supplied random initial
  state. H is 96 or 64; head size 128.
- Speedups are measured against FlashKDA's fused CUTLASS forward, executed
  live on every workload. It is also the shipped baseline solution and the
  default candidate, so a bare run scores ~1.0x — your kernel is racing the
  strongest known implementation.
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
  computed against the baseline.
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
- Correctness on every workload is the only hard requirement.

## Workloads

6 packed-varlen workloads, 3 per head count (H=96 and H=64), all with
total_tokens=8192:
- **fixed**: single sequence of 8192 tokens (`cu_seqlens=None`)
- **mixed_varlen**: 6 sequences of lengths [1300, 547, 2048, 963, 271, 3063]
- **uniform_varlen**: 8 sequences of 1024 tokens each

All 6 workloads are classified "large". Values are synthesized from a seeded
RNG with K3-realistic `A_log`/`dt_bias` parameter distributions; initial states
are fp32 Gaussian values scaled by 0.25. Speed is reported as geomean speedup
vs FlashKDA (all workloads). Correctness is checked against the FLA Triton
reference, atol=rtol=5e-2 with at least 99.9% of elements matching per workload.

Development loop: iterate quickly with the fast subset —

    [Original benchmark command omitted.]

which runs 2 representative workloads (H=96 fixed and H=64 fixed). Run the
full set before committing a new best.

Whenever you make real progress: commit your kernel, run the FULL recorded
benchmark (first command in Objective; the first full run may take much
longer while the baseline JIT-compiles — keep the `--timeout 3600`), and
commit `benchmark.csv` + `[original task locator omitted]` in a follow-up commit. Fast-mode
rows are advisory only; only full runs are valid measurements, and they
require a clean committed tree.

## Starting point — pre-seeded champion kernel

`[original task locator omitted]` is already seeded (and committed) with a strong working
implementation: the `pkda` package — a CuTe-DSL m64/m128 tcgen05/TMEM kernel
pair with workload-aware routing, already once optimized for this machine's
GPU (branch fable5-pkda-b300-best-20260810 of kda-proj-reference) — wrapped
by `[original task locator omitted]`. Treat it as your own code: read
`[original task locator omitted]` and the kernel docstrings first; they document
the design, the routing rule, and per-workload behavior.

Your objective is to make it measurably faster on this machine's GPU:

- First establish the starting score: run the FULL benchmark on the seeded
  kernel, commit the results, and run `commit_and_submit` so the seed's
  authoritative score is on record.
- Then profile (ncu and IKET, in separate runs) and attack the concrete
  bottlenecks. Incremental tuning, structural changes to the pkda kernels,
  and — if profiling justifies it — partial rewrites are all in scope.
- All original rules above still apply: CuTe-DSL only, correctness on every
  workload, benchmark.csv + [original workspace locator omitted] record keeping.
