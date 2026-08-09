// Extracted from sources/blogs/tcgen05-tutorial.md by scripts/extract_blog_code.py
// Heading: ## Key Code > ### Basic tcgen05.mma kernel (17% of peak)
// Original fence language: cuda
// See artifacts/blogs/tcgen05-tutorial/code/PROVENANCE.yaml for origin + license metadata.

// The naive building block: one-thread-launched tcgen05.mma into TMEM.
// ~255 TFLOPS on B200 (17% of peak).
__shared__ uint32_t tmem;
// tcgen05.alloc is .sync.aligned: one whole warp must issue it for cta_group::1,
// so the guard selects a warp, not a single lane. Only the MMA below is
// genuinely single-threaded.
if (threadIdx.x < 32) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;\n"
                 :: "r"(smem_addr));
}
__syncthreads();

for (int k = 0; k < K; k += K_TILE) {
    cp_async(smem_a, A + k);
    cp_async(smem_b, B + k);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    // disable-output-lane is an optional 4-element vector for cta_group::1;
    // all-zero, like omitting it, keeps every output lane enabled.
    // enable-input-d is a predicate, hence the setp.
    uint32_t mask[4] = {0, 0, 0, 0};
    if (threadIdx.x == 0) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n"
                     "}\n"
                     :: "r"(tmem), "l"(desc_a), "l"(desc_b), "r"(0), "r"(1),
                        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
    }
}
