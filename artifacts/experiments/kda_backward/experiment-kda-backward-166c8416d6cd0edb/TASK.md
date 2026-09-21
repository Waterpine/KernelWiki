<!-- Local transcription: operational source-workspace locators were omitted. -->
<!-- Task family: kda_backward; original English title: Optimize the Kimi Delta Attention backward kernel on NVIDIA Blackwell (CuTe-DSL) -->

# Optimize the Kimi Delta Attention backward kernel on NVIDIA Blackwell (CuTe-DSL)

## Objective

Implement a Kimi Delta Attention (KDA, chunked per-channel-gated delta rule)
training backward kernel in CuTe-DSL and make it as fast as possible on the
FULL benchmark workload set, with every workload passing correctness.

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
  `run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens, grad_out)`
  returning `(dq, dk, dv, dg, dbeta, dA_log, ddt_bias)` — the kda_forward
  inputs plus the upstream gradient. Packed varlen format (B=1, T=total
  tokens): bf16 `q/k/v/g/grad_out [1, T, H, 128]`, bf16 beta logits
  `beta [1, T, H]`, fp32 `A_log [H]`, fp32 flattened `dt_bias [H*128]`, python
  float `scale` (1/sqrt(128)), fp32 `initial_state [num_seqs, H, 128, 128]`,
  and `cu_seqlens` (int64 tensor `[num_seqs+1]` of cumulative sequence
  start/end offsets, or `None` for a single sequence). Gradients are bf16
  except fp32 `dA_log` and `ddt_bias`. It may import sibling files inside
  `[original task locator omitted]`.
- `[original task locator omitted]` shows the exact interface by
  wrapping the baseline solution. You may read it (and
  `[original task locator omitted]`) to learn the interface
  and semantics, but your kernel must not call it, nor any prebuilt kernel.
- The op: the backward pass of one tensor-parallel rank of Kimi-K3
  linear-attention training — the chunked delta rule with per-channel
  (fine-grained) gating, differentiated end to end through the in-kernel q/k
  L2 norm, the bounded gate `-5.0 * sigmoid(exp(A_log) * (g + dt_bias))` and
  `beta = sigmoid(beta)` (V-first recurrent-state layout). Each sequence
  starts from its supplied random initial state, which is a constant: no
  initial-state gradient and no final-state output or gradient. The forward
  may be recomputed inside the op — activation recomputation is the
  production default, and its cost is part of the timed work for baseline and
  candidate alike. H is 96 or 64; head size 128.
- Speedups are measured against FLA's native Triton `chunk_kda` autograd
  path executed live on every workload. It is also the shipped baseline
  solution and the default candidate, so a bare run scores ~1.0x.
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
  gated delta rule, chunked scan, attention backward, BF16 attention, TMA,
  TMEM, and `tcgen05`.
- Use ncu-report-skill when profiling or interpreting Nsight Compute reports.
- Proactively use both Nsight Compute and CuTe-DSL IKET, in separate profiling
  runs, to find concrete bottlenecks and drive further kernel improvements.
- Correctness on every workload is the only hard requirement.

## Workloads

6 packed-varlen workloads — the same set as kda_forward — 3 per head count
(H=96 and H=64), all with total_tokens=8192:
- **fixed**: single sequence of 8192 tokens (`cu_seqlens=None`)
- **mixed_varlen**: 6 sequences of lengths [1300, 547, 2048, 963, 271, 3063]
- **uniform_varlen**: 8 sequences of 1024 tokens each

All 6 workloads are classified "large". Values are synthesized from a seeded
RNG with K3-realistic `A_log`/`dt_bias` parameter distributions; initial states
are fp32 Gaussian values scaled by 0.25. Speed is reported as geomean speedup
vs the baseline (all workloads). Correctness is checked against the FLA Triton
autograd reference, atol=rtol=1e-1 with at least 99.5% of elements matching
per workload.

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
