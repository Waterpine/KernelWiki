<!-- Local transcription: operational source-workspace locators were omitted. -->
<!-- Task family: dsa_attention; original English title: Optimize the DeepSeek-V3.2 sparse MLA decode (DSA) kernel on NVIDIA B300 (CuTe-DSL) -->

# Optimize the DeepSeek-V3.2 sparse MLA decode (DSA) kernel on NVIDIA B300 (CuTe-DSL)

## Objective

Implement a DSA sparse MLA decode attention kernel in CuTe-DSL and make it as
fast as possible on the FULL benchmark workload set, with every workload
passing correctness. There is no fixed speedup target and no acceptance
threshold: correctness is the only hard requirement, and speed is optimized
continuously with no finish line.

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
- Actively evaluate and use as many relevant B200 and CUDA 13.2 features as
  possible, including TMA, TMEM, `tcgen05`, warp specialization, persistent
  scheduling, wide vectorized memory operations, and coalesced memory access
  when they fit the kernel.
- Use KernelWiki for research on Blackwell/B200, CUDA 13.2, Triton, sparse
  attention, MLA/DSA, paged KV cache access, BF16 attention, softmax/LSE,
  TMA, TMEM, and `tcgen05`.
- Use ncu-report-skill when profiling or interpreting Nsight Compute reports.
- Proactively use both Nsight Compute and CuTe-DSL IKET, in separate profiling
  runs, to find concrete bottlenecks and drive further kernel improvements.
- There is no fixed speedup target; keep optimizing to push the speedup higher.
  Correctness on every workload is the only hard requirement.

## Workloads

23 official workloads (num_tokens 1..8, 8462 pages, REAL captured
sparse_indices — most tokens have only tens to a few hundred valid (non -1)
indices, so the per-token work is tiny) plus 7 large synthetic workloads
(num_tokens 8..256, up to 32768 pages, dense random indices). The baseline is
timed live and is also the correctness oracle (atol=rtol=1e-2, at least 99.9%
of elements must match). The baseline is a general-purpose MLA decode kernel
and carries substantial fixed launch/prepare cost at these tiny batch sizes —
that is what makes a large mean speedup plausible; the mean is dominated by the
tiny official workloads. The baseline's timed cost includes combining the
separate compressed-KV and RoPE caches into the layout its kernel requires, so
a candidate that consumes the native separate caches directly is credited for
avoiding that work.

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
