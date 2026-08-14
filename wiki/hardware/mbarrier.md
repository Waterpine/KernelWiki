---
id: hw-mbarrier
title: "mbarrier (Memory Barrier Primitives)"
type: hardware
architectures: [sm100, sm100a, sm90, sm90a]
tags: [mbarrier, tma]
confidence: verified
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
related: [hw-tma, hw-tcgen05-mma, technique-warp-specialization, technique-pipeline-stages]
sources: [doc-ptx-isa-sm100, pr-cutlass-2139, blog-tcgen05-tutorial, doc-nvidia-tuning-guide]
aliases: [mbarrier, "memory barrier", "mbar"]
blackwell_relevance: "mbarrier is the primary synchronization primitive between TMA producers and tcgen05 consumers on Blackwell. Phase/parity tracking critical for pipelined kernels."
---

# mbarrier

## Overview

mbarriers are 64-bit shared-memory barrier objects used for phase-based producer/consumer synchronization. TMA and `tcgen05.commit` can signal completion through an mbarrier; a bare `tcgen05.mma` does not automatically arrive on one.

## Key Operations

```ptx
// Initialize: set expected arrival count
mbarrier.init.shared.b64 [mbar_addr], num_arrivals;

// Producer: arrive on barrier (decrements expected count)
mbarrier.arrive.shared.b64 _, [mbar_addr];

// Producer (with byte expectation): used by TMA
mbarrier.arrive.expect_tx.shared.b64 _, [mbar_addr], expected_bytes;
// After this, TMA hardware completes the transaction by arriving with
// the transferred byte count, and the mbarrier flips parity.

// Consumer: wait for parity flip
mbarrier.try_wait.parity.shared.b64 p, [mbar_addr], phase;
@!p bra WAIT_LOOP;
```

## Phase/Parity Semantics

mbarriers have a 1-bit **phase** that flips each time the arrival count reaches zero. Consumers track their expected phase and wait for it:

```cuda
int phase[NUM_STAGES] = {0};
for (int k = 0; k < num_iterations; k++) {
    int stage = k % NUM_STAGES;

    // Wait for this stage's producer to complete
    mbarrier_wait_parity(&mbar[stage], phase[stage]);
    phase[stage] ^= 1;  // Flip when this particular stage slot is reused

    // Use data...
    consume_data(stage);

    // Arrive on next-stage barrier when done
    mbarrier_arrive(&buffer_free[stage]);
}
```

## TMA Integration

TMA directly signals mbarriers when transfer completes, avoiding manual polling:

```cuda
// Producer warp issues TMA with mbarrier target
if (lane_id == 0) {
    uint32_t tx_bytes = TILE_A_BYTES + TILE_B_BYTES;
    mbarrier_arrive_expect_tx(&mbar[stage], tx_bytes);

    // TMA completion will fire the mbarrier automatically
    cp_async_bulk_tensor_2d(smem_A[stage], global_desc, x, y, &mbar[stage]);
    cp_async_bulk_tensor_2d(smem_B[stage], global_desc, x, y, &mbar[stage]);
}
// Do not add an unrelated arrival that advances this phase early.
```

## Pitfalls

1. **Missing phase flip**: Reusing a stage slot without flipping parity causes stale arrivals to satisfy the new wait
2. **Mismatched init count**: If consumer expects N arrivals but producers only arrive M times, barrier never fires
3. **Wrong completion source**: TMA uses the copy's complete-tx mechanism; tcgen05 MMA completion requires `tcgen05.commit` targeting the barrier
4. **Extra arrival**: an arrival not included in the designed arrival/transaction counts can advance a phase early
5. **Wrong scope/state space**: CTA-local and cluster-shared protocols have different address, scope, and proxy-fence requirements

## Related
- [TMA](tma.md) — Primary producer for mbarriers
- [warp-specialization](../techniques/warp-specialization.md) — Uses mbarriers for warp role handoff
