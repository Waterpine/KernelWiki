// Extracted from sources/blogs/deepgemm.md by scripts/extract_blog_code.py
// Heading: ## Key Code > ### SM100 path — tcgen05.mma with UE8M0 block scaling
// Original fence language: cpp
// See artifacts/blogs/deepgemm/code/PROVENANCE.yaml for origin + license metadata.

// On Blackwell, tcgen05.mma consumes UE8M0 scale factors directly.
// 4 UE8M0 values pack into a single uint32; TMEM accumulates in full FP32
// precision so no CUDA-core promotion is needed.
uint32_t packed_scales = pack_ue8m0(sf[0], sf[1], sf[2], sf[3]);
// enable-input-d is a .pred operand, so it cannot be an integer immediate or a
// plain .b32 register: materialise it with setp inside the asm block.
asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %6, 0;\n"
    "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
    "[%0], %1, %2, %3, [%4], [%5], p;\n"
    "}\n"
    :: "r"(tmem_acc), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(sfa_tmem_addr), "r"(sfb_tmem_addr), "r"(1));
