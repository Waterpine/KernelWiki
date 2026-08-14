---
id: migration-wgmma-to-tcgen05
title: "Migrating from wgmma to tcgen05"
type: migration
from_arch: sm90
to_arch: sm100
tags: [tcgen05, wgmma, tmem]
related: [hw-tcgen05-mma, hw-tmem, technique-warp-specialization]
sources: [doc-ptx-isa-sm100, doc-nvidia-tuning-guide, pr-cutlass-2139, blog-tcgen05-tutorial, blog-colfax-cutlass]
blackwell_relevance: "The migration changes the MMA issue contract, accumulator storage, completion protocol, layouts, and epilogue path."
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
reproducibility: pseudocode
---

# Migrating from `wgmma` to `tcgen05`

## Do not translate instruction-for-instruction

Hopper `wgmma.mma_async` and Blackwell `tcgen05.mma` use different accumulator
storage and completion mechanisms. Port the whole mainloop/epilogue contract,
preferably by selecting an SM100 CUTLASS/CuTe collective, rather than replacing
one inline-PTX string.

| Concern | Hopper `wgmma` | Blackwell `tcgen05` |
|---|---|---|
| Hardware issue | warpgroup collective | elected single-thread issue; framework call may be warp-uniform |
| Destination | distributed register fragment | TMEM |
| Operand A | registers or SMEM descriptor, depending on form | SMEM descriptor or TMEM, depending on form |
| Operand B | SMEM descriptor | SMEM descriptor |
| Completion | `wgmma.commit_group` / `wgmma.wait_group` | `tcgen05.commit` + mbarrier wait; tcgen05 loads/stores have their own waits |
| Layouts | WGMMA-specific validity rules | tcgen05 kind/type/major/swizzle validity tables |

## Migration checklist

1. Choose an SM100 MMA kind and legal M/N/K/layout combination.
2. Allocate TMEM collectively in a power-of-two column count (32-column unit).
3. Build A/B layouts and descriptors that exactly match the chosen MMA.
4. Keep TMA completion separate from MMA completion.
5. Commit MMA completion to an mbarrier and wait before dependent TMEM access.
6. Use `tcgen05.fence` only for the documented cross-thread ordering role.
7. Read TMEM with a legal collective load and `tcgen05.wait::ld`.
8. Explicitly deallocate TMEM before exit.
9. Re-tune warps, stages, cluster shape, and tile sizes; do not carry Hopper
   choices forward as constants.

## Schematic SM100 lifecycle

```text
issuing warp:  tcgen05.alloc -> load returned taddr
producer:      TMA to SMEM -> wait for TMA mbarrier phase
MMA issuer:    tcgen05.mma -> tcgen05.commit(mma_done)
consumer:      wait mma_done -> fence::after_thread_sync -> tcgen05.ld
consumer:      tcgen05.wait::ld -> epilogue/store
issuing warp:  tcgen05.dealloc -> relinquish_alloc_permit
```

## Swizzling correction

Changing every Hopper layout to 128-byte swizzling is not a migration rule.
PTX supports all swizzle modes for common K-major tcgen05 operand layouts and
restricts some transposed/type combinations. Select the mode from the exact
validity table and ensure the TMA producer and MMA descriptor agree.

## Two-CTA mode

`.cta_group::2` does not automatically make an existing kernel faster. It adds
peer-CTA allocation, descriptor, synchronization, cluster-launch, and
deallocation constraints. Use it when the supported larger M tile and reuse
outweigh the extra coordination and occupancy cost.

## Sources

- [PTX ISA: WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions)
- [PTX ISA: tcgen05](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-family-instructions)
- [CUTLASS tcgen05 programming guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)
