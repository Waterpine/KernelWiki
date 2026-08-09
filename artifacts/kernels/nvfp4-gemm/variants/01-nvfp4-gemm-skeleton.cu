// provenance: derived from pr-cutlass-2139, kernel-nvfp4-gemm, hw-nvfp4; not upstream code
// origin: wiki/kernels/nvfp4-gemm.md Phase 3 variant

// Minimal NVFP4 GEMM skeleton. Packs 2x FP4 per byte; scale factors
// are UE4M3 (FP8 E4M3) per 16-element block. tcgen05.mma handles both.

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdint>

constexpr int BLOCK_SCALE = 16;  // NVFP4 scale granularity

template <int TILE_M, int TILE_N, int TILE_K>
__device__ void nvfp4_gemm_tile(
    uint32_t tmem_acc,
    const __nv_fp4_e2m1* A, const __nv_fp4_e2m1* B,
    const __nv_fp8_e4m3* SFA, const __nv_fp8_e4m3* SFB,
    int M, int N, int K)
{
    // Launch with tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X:
    // .kind::f8f6f4 is not a block-scale kind, and E2M1 operands with UE4M3
    // 16-element scales require .scale_vec::4X (alias .block16).
    // A and B descriptors address the operand tiles; the scale factors live in
    // TMEM and are addressed separately by [scale-A-tmem] / [scale-B-tmem].
    if (threadIdx.x == 0) {
        uint64_t desc_a = make_nvfp4_desc(A);
        uint64_t desc_b = make_nvfp4_desc(B);
        uint32_t sfa_tmem = tmem_scale_a();  // SFA staged into dedicated TMEM columns
        uint32_t sfb_tmem = tmem_scale_b();  // SFB likewise
        uint32_t idesc = make_instruction_desc<TILE_M, TILE_N, TILE_K>();
        // enable-input-d is a .pred operand, so it is derived with setp
        asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
            "[%0], %1, %2, %3, [%4], [%5], p;\n\t}\n"
            :: "r"(tmem_acc), "l"(desc_a), "l"(desc_b), "r"(idesc),
               "r"(sfa_tmem), "r"(sfb_tmem), "r"(1));
    }
}
