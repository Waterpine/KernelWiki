// Extracted from sources/blogs/flash-attention-4.md by scripts/extract_blog_code.py
// Heading: ## Key Code > ### 2-CTA cooperative backward
// Original fence language: cuda
// See artifacts/blogs/flash-attention-4/code/PROVENANCE.yaml for origin + license metadata.

// 2-CTA cooperative backward: paired CTAs in a cluster share a single TMEM
// accumulator half, halving SMEM traffic for dK/dV accumulation.
// cta_group::2 takes an 8-element disable-output-lane vector;
// enable-input-d is a predicate.
uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %4, 0;\n"
    "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8, %9, %10, %11, %12}, p;\n"
    "}\n"
    : : "r"(tmem_acc_shared), "l"(desc_a), "l"(desc_b), "r"(0), "r"(1),
        "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
        "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
