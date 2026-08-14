---
version_sensitive:
  id: vs-triton-3.6-blackwell-tcgen05
---

# Worked Query Examples

These examples show navigation paths. Treat the linked source record—not the query page—as the authority for an instruction spelling, benchmark, or version claim.

## Fast GEMM on B200

1. Query `gemm` pages by architecture.
2. Select the data format: FP8 block scale, NVFP4, or another documented contract.
3. Read the matching kernel and hardware pages.
4. Follow every performance claim to its pinned source and shape.

```bash
python3 scripts/query.py --type kernel --tag gemm --architecture sm100
```

The community tcgen05 tutorial records a progression for its own GEMM shapes. It is useful evidence about that implementation, not a canonical sequence of optimization percentages or a universal path to 98%.

## Low SM utilization

```bash
python3 scripts/query.py --symptom low-sm-utilization
python3 scripts/get_page.py pattern-low-sm-utilization
```

Use the pattern page to distinguish too little grid work, tail imbalance, dependency stalls, and resource-limited occupancy. CLC can cancel not-yet-started cluster work in a persistent scheduler; it is not a general work queue or a guaranteed utilization increase.

## Find tcgen05 usage in CUTLASS

```bash
python3 scripts/query.py --tag tcgen05 --repo cutlass --limit 30
python3 scripts/grep_wiki.py "tcgen05\\.mma" --only sources
```

The `UMMA` alias resolves to `tcgen05`, but exact instruction forms must be checked against the pinned PTX ISA.

## FlashAttention-4

```bash
python3 scripts/get_page.py kernel-flash-attention-4 --follow-sources
```

The paper reports a maximum of 1613 TFLOP/s on its documented B200 BF16 sweep, about 71% of the theoretical maximum used by the paper. That maximum is not a single-shape guarantee. The page also distinguishes the hybrid software/hardware exponential strategy from a blanket software replacement.

## Hopper WGMMA to Blackwell tcgen05

```bash
python3 scripts/get_page.py migration-wgmma-to-tcgen05
python3 scripts/get_page.py hw-tcgen05-mma
```

Check accumulator placement, issue scope, completion, TMEM allocation, and architecture-specific target compatibility separately; mnemonic substitution alone is not a migration.

## GPU Mode NVFP4 contest

```bash
python3 scripts/query.py --type contest --tag nvfp4
python3 scripts/get_page.py contest-gpumode-p1
```

Participant timings and optimization progressions are source-reported contest results. Preserve the cited shape, harness, and access date and do not convert them into general B200 limits.

## Gated Delta Net on Blackwell

```bash
python3 scripts/query.py "gated delta net decode" --language triton
```

The canonical page explains the recurrence and chunk-boundary dependency. Use its artifact bundle or a pinned upstream implementation; its conceptual pseudocode is not presented as a drop-in Triton kernel.

## Memory-bound kernels

```bash
python3 scripts/query.py --symptom memory-bound
```

Vector width, cache hints, and occupancy are candidates to measure. Participant-reported NVFP4 GEMV timings apply only to their contest configurations.

## FlashInfer PRs for FP8 MoE

```bash
python3 scripts/query.py --repo flashinfer --tag moe --limit 30
python3 scripts/query.py --repo flashinfer --tag fp8 --limit 30
```

## SM100 PTX

```bash
python3 scripts/get_page.py lang-ptx --body-only
python3 scripts/grep_wiki.py "tcgen05" --only wiki --context 0
```

The PTX page separates SM100-specific tcgen05/TMEM/CLC behavior from older instructions that remain usable on SM100.

## Synthesis rules

- Cite the source ID and its pinned version or retrieval date.
- Report every benchmark with GPU, dtype, shape, metric, baseline, and software context available in the source.
- Treat illustrative snippets as pseudocode unless their artifact provenance and build instructions say otherwise.
- Do not transfer SM90 results to SM100 or vice versa without new evidence.
- For a `verified` page, check that `evidence_basis` actually contains the required independent source categories.
