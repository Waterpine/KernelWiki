<!-- Local transcription: operational source-workspace locators were omitted. -->
<!-- Task family: dsa_attention; original English title: Optimize the DeepSeek-V3.2 sparse MLA decode (DSA) kernel on the official contest suite (CuTe-DSL) -->

# Optimize the DeepSeek-V3.2 sparse MLA decode (DSA) kernel on the official contest suite (CuTe-DSL)

## Objective

Implement a DSA sparse MLA decode attention kernel in CuTe-DSL and maximize
the OFFICIAL SCORE on the official workload set: the arithmetic mean of
per-workload speedups vs the flashinfer baseline over all 23 MLSys26-contest
workloads. Any failing workload zeroes the official score, so correctness on
every workload is a hard requirement.

Measure the full workload set from inside `[original task locator omitted]`:

    [Original benchmark command omitted.]

## Scoring and submission

The primary metric is the official score printed at the end of every run —
arithmetic mean speedup over all 23 workloads, 0 if any workload fails.
Geomean speedups are reported as diagnostics.

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
  `run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)` — bf16
  q_nope `[T,16,512]`, q_pe `[T,16,64]`, paged caches `[P,64,512]` /
  `[P,64,64]` (page_size 64), int32 `sparse_indices [T,2048]` of token-level
  indices into the flattened cache with -1 padding; output bf16 `[T,16,512]`.
  It may import sibling files inside `[original task locator omitted]`.
- `[original task locator omitted]` shows the exact interface by wrapping the
  flashinfer baseline (`trtllm_batch_decode_with_kv_cache_mla`,
  sparse_mla_top_k=2048). You may read it to learn the interface and
  semantics, but your kernel must not call it, nor flashinfer, nor any
  prebuilt kernel.
- The op: per query token, gather the valid (non -1) sparse KV entries,
  compute softmax((q_nope @ Kc^T + q_pe @ Kp^T) * sm_scale) @ Kc over them,
  output the 512-dim compressed values per head.
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

The 23 official MLSys26-contest workloads (num_tokens 1..8, 8462 pages, REAL
captured sparse_indices — most tokens have only tens to a few hundred valid
(non -1) indices, so the per-token work is tiny). Every workload is
classified by `size_class`: all 23 are **small** (num_tokens < 16); there are
no large workloads in the official suite, so the large geomean reports n/a
and the launch-overhead regime IS the whole official score.

The baseline is timed live and is also the correctness oracle. The contest
gate applies: atol=rtol=1e-2 and EVERY element must match (stricter than the
internal `dsa_attention` task's 99.9%). The baseline is a general-purpose MLA
decode kernel and carries substantial fixed launch/prepare cost at these tiny
batch sizes — that is what makes a large mean speedup possible. The
baseline's timed cost includes combining the separate compressed-KV and RoPE
caches into the layout its kernel requires, so a candidate that consumes the
native separate caches directly is credited for avoiding that work.

Development loop: iterate quickly with the fast subset —

    [Original benchmark command omitted.]

which runs 4 small workloads (the official suite has no large class). Run
the full set before committing a new best.

Whenever you make real progress: commit your kernel, run the FULL recorded
benchmark (first command in Objective; the first full run may take much
longer while flashinfer JIT-compiles baseline kernels — keep the
`--timeout 3600`), and commit `benchmark.csv` + `[original task locator omitted]` in a
follow-up commit. Fast-mode rows are advisory only; only full runs are valid
measurements, and they require a clean committed tree.

## Starting point — pre-seeded rules-clean kernel

`[original task locator omitted]` is pre-seeded (and committed) with the best
integrity-clean DSA kernel from the previous flame runs (CuTe-DSL). Fresh
full-protocol confirmation on this harness (2026-08-17, B300): official mean
**37.6827x**, 23/23 official workloads, all-geomean 34.05x. Its techniques,
documented in the kernel docstrings: fp32 QK scores, hi/lo split-P BF16 pairs
for exact-P PV MMA with fp32 accumulation; a T=1 head-per-CTA "tiny2"
microkernel; a <=64-row-prefix small-token microkernel with an exact on-device
tail scan and a general-path fallback; split-K partial merge guarded by a
launch-id watermark protocol (stale cross-call flags are provably never
consumed); TMA multicast for q.

Your objective: MAXIMIZE the official score — the mean-speedup headline in
every judge feedback line, zeroed by any failing workload.

## Contest integrity rules (BINDING — a violation zeroes the whole run in review)

Every accepted submission is re-audited after the run (code review + an
empirical A/B/A cross-call test). The kernel must compute its outputs from the
CURRENT call's inputs, on every call:

1. NO cross-call caching of input-derived data. The bench calls the kernel
   repeatedly with the SAME tensors; exploiting that is cheating. Forbidden:
   module dicts / functools caches / persistent buffers keyed on tensor
   contents, `data_ptr()`, `id()`, `_version`, or weakrefs to inputs; caching
   topk-index-derived masks / counts / gather plans / split maps; "compute on
   first call, reuse after" logic. Allowed: compile caches keyed ONLY by
   shapes/dtypes/static config; persistent workspaces IF every consumed
   element is rewritten from live inputs each call (the seed's launch-id
   watermark protocol is the reference pattern — keep it intact).
2. Cross-call and first-call correctness are part of correctness. The kernel
   must be right when inputs CHANGE between calls at the same shape, and on
   the very first call after import. A 38.09x sibling of the seed was REJECTED
   for first-call NaN/mismatch at T=6/7 (it removed the EXP2_CLAMP overflow
   guard and had a first-launch race). Self-test with an A/B/A drill before
   every submission: run inputs A cold, A warm, changed inputs B, then A
   again — all four must match the reference.
3. NO measurement manipulation. The previous 45.75x/49.7x "records" were
   disqualified for: a 48MB carve kept alive to slow the in-process live
   baseline, persisting-L2 cache-policy pinning of the rows selected by the
   CURRENT indices (keyed by data_ptr), and index-content caching. Also
   forbidden: side-stream work escaping the timed region, monkeypatching
   torch/bench modules, harness detection, deferring work past the sync point.
4. Keep robustness guards. Do not remove NaN/overflow clamps or
   input-validity fallbacks merely because the official workloads do not
   trigger them; distribution-informed guard removal is treated as a
   violation in review. Precision floors: fp32 accumulation for QK, softmax,
   and PV stays.

Rules of progress (both agents, every turn):

- First turn only: run the FULL benchmark on the untouched seed, commit
  `[original task locator omitted]`, and `commit_and_submit` it so the seed's authoritative
  official mean is on record.
- A submission is a new best ONLY if it improves the official MEAN by >= 1%
  over the current best. Before claiming a new best, run the FULL suite twice
  and pass the A/B/A drill.
- Maintain `[original task locator omitted]` (committed): one line per accepted submission
  — submission id, MEAN, geomean, plus the current best. Read it at the start
  of every turn; it is the shared state between the alternating agents.
- All 23 official workloads are decode-tiny (num_tokens 1..8): per-call fixed
  costs (launches, host syncs, allocations, descriptor setup) are the lever.
  Shape-keyed specialization is allowed; specialization keyed on input DATA
  patterns must probe the live inputs on-device every call and fall back to a
  general path (as the seed's microkernels do).

All original rules above still apply: CuTe-DSL only, correctness on every
workload, benchmark.csv + [original workspace locator omitted] record keeping, `--fast` for quick
iteration, and the FULL set before every submission.
