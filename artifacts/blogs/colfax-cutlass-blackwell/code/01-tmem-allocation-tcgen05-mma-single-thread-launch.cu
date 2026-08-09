// Extracted from sources/blogs/colfax-cutlass-blackwell.md by scripts/extract_blog_code.py
// Heading: ## Key Code > ### TMEM allocation + tcgen05.mma (single-thread launch)
// Original fence language: cuda
// See artifacts/blogs/colfax-cutlass-blackwell/code/PROVENANCE.yaml for origin + license metadata.

// UMMA on Blackwell: one thread drives the MMA for the whole CTA.
// Accumulator lives in TMEM, not registers.
__shared__ uint32_t tmem_addr;

// tcgen05.alloc is .sync.aligned: with cta_group::1 one whole warp of the CTA
// must issue it, all 32 threads with identical operands. A single-thread guard
// would be undefined behaviour, so this is a warp guard, not a lane guard.
if (threadIdx.x < 32) {
    // Allocate 256 TMEM columns (all 128 lanes of each column) for the accumulator
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_addr));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;\n"
                 :: "r"(smem_addr));
}
__syncthreads();

// Issue UMMA: A and B live in SMEM, C accumulates into TMEM.
// disable-output-lane is an optional 4-element vector for cta_group::1 -- an
// all-zero mask, like omitting the operand, keeps every output lane enabled.
// enable-input-d is a predicate, hence the setp.
uint32_t mask[4] = {0, 0, 0, 0};
if (threadIdx.x == 0) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n"
        "}\n"
        :: "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(0), "r"(1),
           "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
}
