<!-- Local transcription: operational source-workspace locators were omitted. -->
<!-- Task family: moe; original English title: Optimize the FP8 block-scale DeepSeek-V3 MoE kernel on NVIDIA B300 (CuTe-DSL) -->

# Optimize the FP8 block-scale DeepSeek-V3 MoE kernel on NVIDIA B300 (CuTe-DSL)

## Objective

Implement an FP8 block-scale MoE kernel with fused DeepSeek-V3 sigmoid routing
in CuTe-DSL and make it as fast as possible on the FULL benchmark workload set,
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

- `[original task locator omitted]` is a git clone of the kernel workspace. Do all work inside it and
  read its `AGENTS.md` (environment + benchmark policy) before starting.
- Your kernel lives at `[original task locator omitted]` and must expose `run(...)`
  with the signature of `[original task locator omitted]` — routing
  logits/bias, fp8 (e4m3) block-scale (128) hidden states and GEMM weights,
  top_k=8, n_group=8, topk_group=4, 256 global / 32 local experts, hidden
  7168, intermediate 2048, SwiGLU, bf16 output. It may import sibling files
  inside `[original task locator omitted]`.
- `[original task locator omitted]` shows the exact interface by wrapping the
  flashinfer baseline (`trtllm_fp8_block_scale_moe`). You may read it to
  learn the interface and semantics, but your kernel must not call it, nor
  flashinfer, nor any prebuilt kernel.
- The op: fused no-aux-loss DeepSeek-V3/R1 routing (sigmoid scores, group
  top-k) followed by FP8 block-scale grouped GEMM + SwiGLU + down-projection
  and weighted combine.
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
- Keep NCU profiling records for each major optimization direction under
  `[original task locator omitted]` and commit them.
- Actively evaluate and use as many relevant B200 and CUDA 13.2 features as
  possible, including TMA, TMEM, `tcgen05`, warp specialization, persistent
  scheduling, wide vectorized memory operations, and coalesced memory access
  when they fit the kernel.
- Use KernelWiki for research on Blackwell/B200, CUDA 13.2, Triton, sparse
  attention, MLA/DSA, paged KV cache access, BF16 attention, softmax/LSE,
  TMA, TMEM, and `tcgen05`.
- Use ncu-report-skill when profiling or interpreting Nsight Compute reports.
- There is no fixed speedup target; keep optimizing to push the speedup higher.
  Correctness on every workload is the only hard requirement.

## Workloads

19 official workloads (seq_len 1..14107, real captured routing logits/bias)
plus 1 large synthetic workload (seq_len 32768). Official-suite baseline
latencies are frozen B200 measurements; the large workload times the baseline
live. Correctness is checked against the pure-PyTorch reference, atol=0.05,
rtol=0.15, at least 99% of elements must match, and whole-tensor relative L2
error must stay below 0.25 — a genuinely computed FP8 kernel passes these
with several-fold margin; approximations that trade correctness for speed
will not.

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
