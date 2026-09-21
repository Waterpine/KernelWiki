<!-- Local transcription: operational source-workspace locators were omitted. -->
<!-- Task family: moe; original English title: Optimize the FP8 block-scale DeepSeek-V3 MoE kernel on the official contest suite (CuTe-DSL) -->

# Optimize the FP8 block-scale DeepSeek-V3 MoE kernel on the official contest suite (CuTe-DSL)

## Objective

Implement an FP8 block-scale MoE kernel with fused DeepSeek-V3 sigmoid routing
in CuTe-DSL and maximize the OFFICIAL SCORE on the official workload set: the
arithmetic mean of per-workload speedups vs the flashinfer baseline over all
19 MLSys26-contest workloads, judged under the contest acceptance gate. Any
failing workload zeroes the official score, so correctness on every workload
is a hard requirement.

Measure the full workload set from inside `[original task locator omitted]`:

    [Original benchmark command omitted.]

## Scoring and submission

The primary metric is the official score printed at the end of every run —
arithmetic mean speedup over all 19 workloads, 0 if any workload fails.
Geomean speedups (all / large / small) are reported as diagnostics.

After every change that measurably improves performance, from inside `[original task locator omitted]`
run:

    [Original benchmark command omitted.]

to commit the progress and submit the kernel for authoritative out-of-docker
scoring (it submits only when `[original task locator omitted]` changed and a local smoke check
passes).

## Layout

- `[original task locator omitted]` is a fresh git repository generated from KDA's task recipe. Do all work inside it and
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
- Nsight Compute hardware-counter profiling is enabled through the GPU broker;
  run it as `gpu-run -- ncu ...` when profiling optimization hypotheses.

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
- Actively evaluate and use as many relevant Blackwell and CUDA 13.2 features as
  possible, including TMA, TMEM, `tcgen05`, warp specialization, persistent
  scheduling, wide vectorized memory operations, and coalesced memory access
  when they fit the kernel.
- Use KernelWiki for research on Blackwell, CUDA 13.2, Triton, sparse
  attention, MLA/DSA, paged KV cache access, BF16 attention, softmax/LSE,
  TMA, TMEM, and `tcgen05`.
- Use ncu-report-skill when profiling or interpreting Nsight Compute reports.
- Proactively use both Nsight Compute and CuTe-DSL IKET, in separate profiling
  runs, to find concrete bottlenecks and drive further kernel improvements.
- Correctness on every workload is the only hard requirement — one failing
  workload zeroes the official score.

## Workloads

The 19 official MLSys26-contest workloads (real captured routing logits and
bias; seq_len 1..14107). Every workload is classified by `size_class`:

- **3 large** (`seq_len >= 512`): seq_len 901, 11948, 14107 — grouped-GEMM
  throughput dominates.
- **16 small** (`seq_len < 512`): seq_len 1..80 — routing + launch overhead
  dominates, and these workloads dominate the arithmetic-mean official score.

Every workload times the flashinfer baseline live. Correctness is checked
against the pure-PyTorch reference under the CONTEST acceptance gate:
atol=1.0, rtol=0.3, at least 90% of elements must match, and whole-tensor
relative L2 error must stay below 0.25. The gate is looser per element than
the internal `moe` task, but it is unforgiving in aggregate: one badly
mis-routed token on a small workload (e.g. 1 of 7 tokens = 14% of elements)
fails that workload and zeroes the official score. Routing must therefore be
exact — compute sigmoid/group-top-k in fp32 and match the reference's
`torch.topk` tie-breaking (lowest index wins on equal scores); real captured
logits contain near-ties.

Development loop: iterate quickly with the fast subset —

    [Original benchmark command omitted.]

which runs a size-balanced dev subset — 2 small + 2 large workloads — so a
quick iteration exercises both regimes. Run the full set before committing a
new best.

Whenever you make real progress: commit your kernel, run the FULL recorded
benchmark (first command in Objective; the first full run may take much
longer while flashinfer JIT-compiles baseline kernels — keep the
`--timeout 3600`), and commit `benchmark.csv` + `[original task locator omitted]` in a
follow-up commit. Fast-mode rows are advisory only; only full runs are valid
measurements, and they require a clean committed tree.

## Starting point — pre-seeded rules-clean champion kernel

`[original task locator omitted]` is pre-seeded (and committed) with the best integrity-clean
MoE kernel: the archived FP8 block-scale champion (CuTe-DSL: `kernel.py` plus
`moe_*.py` siblings — fused DeepSeek-V3 routing, grouped GEMM1/SwiGLU/GEMM2).
Fresh full-protocol confirmation on this harness (2026-08-17, B300): official
mean **2.2964x**, 19/19 official workloads, large geomean 2.6730x, small
geomean 2.1731x. Weights are requantized ONCE on the host (cached by weight
tensor identity) onto a per-128-column whole-K e4m3 grid — this stays fp8 and
is allowed, but do NOT go below e4m3 anywhere (see integrity rule 1).

`[original task locator omitted]` (committed) is REFERENCE MATERIAL,
not the seed: a fully-clean single-file TRITON kernel (weights consumed as raw
e4m3 with native-granularity fp32 block scales every call, zero
preprocessing). On this harness it scores 1.94-2.00x mean (19/19; large
geomean only ~1.47x), so it loses to the seed overall — but its techniques
are worth mining: PDL (griddepcontrol) chaining of routing/GEMM1/requant/
GEMM2, a persistent fused kernel for T<=128 with in-kernel spin barriers and
a trailing-cleanup workspace protocol, split-K "eco mode" picked in-kernel
from the CURRENT call's routing counts.

LANGUAGE RULE ADJUSTMENT for this run: CuTe-DSL remains the primary language
(the seed is CuTe), but Triton is also ALLOWED (this supersedes the "CuTe-DSL
only" rule above; the contest itself accepts any language) — e.g. for a
hybrid where a Triton small-T path beats the CuTe one. Only adopt a rewrite
if it beats the current best end-to-end under the same integrity rules. Note
IKET profiling is CuTe-only; use Nsight Compute for Triton paths.

Your objective: MAXIMIZE the official score — the mean-speedup headline in
every judge feedback line, zeroed by any failing workload.

## Contest integrity rules (BINDING — a violation zeroes the whole run in review)

Every accepted submission is re-audited after the run (code review + an
empirical A/B/A cross-call test). The kernel must compute its outputs from the
CURRENT call's inputs, on every call:

1. NO precision reduction of the provided weights below fp8. w13/w2 arrive as
   float8_e4m3fn with fp32 block scales and must stay >= e4m3 end-to-end:
   requantizing weights to E2M1/FP4, E2M3/FP6, int4, or any sub-8-bit format
   is a violation — the previous 3.68x "best" was disqualified exactly for
   E2M1 weights. The seed's whole-K e4m3 rescale is the accepted ceiling of
   aggressiveness; document any further scaling-grid change. Dynamic fp8
   e4m3 requantization of the INTERMEDIATE activation (per row/128-block) is
   fine — the seed and the official baseline both do it.
2. NO cross-call caching of input-derived data. The bench calls the kernel
   repeatedly with the SAME tensors; exploiting that is cheating. Forbidden:
   module dicts / functools caches keyed on tensor contents; caching routing
   results or activation-derived stats across calls. Allowed: compile/launch
   caches keyed ONLY by shapes/dtypes/static config; the seed's one-time
   WEIGHT-tensor preprocessing cached by weight identity (weights are static
   in real serving) — keep it weights-only; persistent workspaces IF every
   consumed element is rewritten from live inputs each call. Cautionary tale:
   a whole 3.0-3.2x submission lineage from the previous flame was
   disqualified because a tiny-T path used 64-row m-tiles while the routing
   kernel re-zeroed only META_P//128 watermark entries — GEMM2 consumed the
   PREVIOUS call's activations and the judge never noticed (it feeds
   identical inputs per workload). If you touch tile sizes, re-derive every
   watermark/reset width.
3. Cross-call and first-call correctness are part of correctness. The kernel
   must be right when inputs CHANGE between calls at the same shape, and on
   the very first call after import. Self-test with an A/B/A drill before
   every submission: inputs A cold, A warm, changed inputs B (different
   hidden/logits, same weights), then A again — all must match the reference.
4. NO measurement manipulation: no allocations kept alive to slow the
   in-process live baseline, no persisting-L2 pinning keyed on input data, no
   side-stream work escaping the timed region, no monkeypatching torch/bench,
   no harness detection. Genuine, actively-used workspace is fine.
5. Correctness margin is part of the deliverable. The contest gate (atol=1,
   rtol=0.3, >=90% matched, whole-tensor rel-L2 < 0.25) is zero-on-fail;
   keep sigmoid / group-top-k routing scoring in fp32 with the reference's
   `torch.topk` tie-breaking (lowest index wins), and never trade gate margin
   for speed. If a change visibly shrinks matched-ratio or error margins,
   treat it as a regression even when it still passes.

Rules of progress (both agents, every turn):

- First turn only: run the FULL benchmark on the untouched seed, commit
  `[original task locator omitted]`, and `commit_and_submit` it so the seed's authoritative
  MEAN (official score) and LARGE (large geomean) are on record. LARGE minus
  1% is a HARD FLOOR for the rest of the run.
- A submission is a new best ONLY if it improves MEAN by >= 1% over the
  current best while LARGE stays above its floor. Before claiming a new best,
  run the FULL suite twice and pass the A/B/A drill.
- Maintain `[original task locator omitted]` (committed): one line per accepted submission
  — submission id, MEAN, LARGE, small geomean — plus the current best and the
  LARGE floor. Read it at the start of every turn; it is the shared state
  between the alternating agents.
- Where the headroom is: 16 of the 19 workloads have seq_len 1..80 — routing
  and launch overhead dominated — and they dominate the arithmetic mean; the
  3 large workloads (seq_len 901/11948/14107) protect the throughput regime
  via the LARGE floor. The seed's small geomean (2.17x) trails its large
  (2.67x), so the mean moves on the small side: per-call fixed costs (kernel
  launches, host syncs, allocations, descriptor setup) are the lever.

All other original rules above still apply: correctness on every workload,
benchmark.csv + [original workspace locator omitted] record keeping, `--fast` for quick iteration, and
the FULL set before every submission.
