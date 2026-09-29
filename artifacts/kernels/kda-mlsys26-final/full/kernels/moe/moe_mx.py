"""Weight preprocessing for the MXFP8 path.

Decomposes each per-128x128-block f32 weight scale s into s = r * 2^e with
r in (0.5, 1]: r is folded into the fp8 weight values (one requantization
rounding) and e is emitted as UE8M0 bytes in the tcgen05 scale-factor atom
layout (per 32-K-element granularity: the per-128 byte is replicated 4x along
K; all 128 rows of an N-atom share the byte).

SF atom byte address (K-major, sf_vec 32), for row r, scale block kb, rep k1:
  (r // 128) * (KB * 512) + kb * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + k1

Run once per distinct weight set and cached host-side (see kernel.py).
"""

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32

F8 = cutlass.Float8E4M3FN
F32 = cutlass.Float32

XFORM_CTAS = 1184
XFORM_THREADS = 256


@cute.kernel
def prep_scales_kernel(
    scale: cute.Tensor,   # (E*NB*KB,) f32 flat
    rr: cute.Tensor,      # (E*NB*KB,) f32: residual in (0.5, 1]
    byt: cute.Tensor,     # (E*NB*KB,) i32: ue8m0 byte
    n_items: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    i = bidx * XFORM_THREADS + tidx
    step = gdim * XFORM_THREADS
    tb = cute.make_rmem_tensor(cute.make_layout(1), F32)
    ib = cute.make_tensor(
        cute.recast_ptr(tb.iterator, dtype=Int32), cute.make_layout(1)
    )
    while i < n_items:
        s = scale[i]
        tb[0] = s
        bits = ib[0]
        eb = (bits >> 23) & 255
        if (bits & 8388607) != 0:
            eb = eb + 1
        if eb > 254:
            eb = Int32(254)
        # r = s * 2^(127 - eb), exact pow2 built from bits
        ib[0] = (254 - eb) << 23
        r = s * tb[0]
        rr[i] = r
        byt[i] = eb
        i += step


@cute.kernel
def fold13_kernel(
    w_in: cute.Tensor,    # (32*4096*7168,) fp8 flat, row-major
    rr: cute.Tensor,      # (32*32*56,) f32
    w_out: cute.Tensor,   # tile-contiguous: (e, j16, kt56, h2, i128, k128)
    n_items: Int32,       # 32*4096*112 (64B output chunks)
):
    """Residual fold + reorder into tile-contiguous layout so each GEMM1
    k-tile B pull is one sequential 32KB DRAM burst."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    i = bidx * XFORM_THREADS + tidx
    step = gdim * XFORM_THREADS
    while i < n_items:
        i2 = i
        kc = i2 & 1
        i2 = i2 >> 1
        ii = i2 & 127
        i2 = i2 >> 7
        h = i2 & 1
        i2 = i2 >> 1
        kt = i2 % 56
        i2 = i2 // 56
        j = i2 & 15
        e = i2 >> 4
        r_row = ii + 128 * j + 2048 * h
        nb = j + 16 * h
        sidx = (e * 32 + nb) * 56 + kt
        r = rr[sidx]
        src_off = (e * 4096 + r_row) * 7168 + kt * 128 + kc * 64
        src = cute.make_tensor(
            (w_in.iterator + src_off).align(16), cute.make_layout(64)
        )
        frag = cute.make_rmem_tensor(cute.make_layout(64), F8)
        cute.autovec_copy(src, frag)
        v = frag.load().to(F32) * r
        frag.store(v.to(F8))
        dst = cute.make_tensor(
            (w_out.iterator + i * 64).align(16), cute.make_layout(64)
        )
        cute.autovec_copy(frag, dst)
        i += step


@cute.kernel
def fold2_kernel(
    w_in: cute.Tensor,    # (32*7168*2048,) fp8 flat, row-major
    rr: cute.Tensor,      # (32*56*16,) f32
    w_out: cute.Tensor,   # tile-contiguous: (e, nt28, kt16, r256, k128)
    n_items: Int32,       # 32*7168*32 (64B output chunks)
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    i = bidx * XFORM_THREADS + tidx
    step = gdim * XFORM_THREADS
    while i < n_items:
        i2 = i
        kc = i2 & 1
        i2 = i2 >> 1
        ri = i2 & 255
        i2 = i2 >> 8
        kt = i2 & 15
        i2 = i2 >> 4
        nt = i2 % 28
        e = i2 // 28
        r_row = nt * 256 + ri
        nb = (r_row >> 7)
        sidx = (e * 56 + nb) * 16 + kt
        r = rr[sidx]
        src_off = (e * 7168 + r_row) * 2048 + kt * 128 + kc * 64
        src = cute.make_tensor(
            (w_in.iterator + src_off).align(16), cute.make_layout(64)
        )
        frag = cute.make_rmem_tensor(cute.make_layout(64), F8)
        cute.autovec_copy(src, frag)
        v = frag.load().to(F32) * r
        frag.store(v.to(F8))
        dst = cute.make_tensor(
            (w_out.iterator + i * 64).align(16), cute.make_layout(64)
        )
        cute.autovec_copy(frag, dst)
        i += step


@cute.kernel
def expand_sf_atoms_kernel(
    byt: cute.Tensor,     # (E*NB*KB,) i32 ue8m0 bytes (NB = N/128 atoms)
    sf: cute.Tensor,      # (E*NB*KB*512,) u8 atom-packed output
    n_atoms: Int32,       # E*NB*KB
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    i = bidx * XFORM_THREADS + tidx
    step = gdim * XFORM_THREADS
    frag = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int128)
    while i < n_atoms:
        b = byt[i]
        w = b | (b << 8) | (b << 16) | (b << 24)
        p32 = cute.make_tensor(
            cute.recast_ptr(frag.iterator, dtype=Int32), cute.make_layout(16)
        )
        for j in cutlass.range_constexpr(16):
            p32[j] = w
        for half in cutlass.range_constexpr(8):
            dst = cute.make_tensor(
                cute.recast_ptr(
                    sf.iterator + (i * 512 + half * 64), dtype=cutlass.Int128
                ).align(16),
                cute.make_layout(4),
            )
            cute.autovec_copy(frag, dst)
        i += step


@cute.jit
def transform_weights(
    w13: cute.Tensor,      # (32*4096*7168,) fp8 flat
    w13_s: cute.Tensor,    # (32*32*56,) f32 flat
    w13f: cute.Tensor,
    sfb13: cute.Tensor,    # (32*32*56*512,) u8
    rr13: cute.Tensor,     # (32*32*56,) f32 scratch
    by13: cute.Tensor,     # (32*32*56,) i32 scratch
    w2: cute.Tensor,       # (32*7168*2048,) fp8 flat
    w2_s: cute.Tensor,     # (32*56*16,) f32 flat
    w2f: cute.Tensor,
    sfb2: cute.Tensor,     # (32*56*16*512,) u8
    rr2: cute.Tensor,
    by2: cute.Tensor,
    stream: cuda_driver.CUstream,
):
    n13 = 32 * 32 * 56
    prep_scales_kernel(w13_s, rr13, by13, Int32(n13)).launch(
        grid=(64, 1, 1), block=(XFORM_THREADS, 1, 1), stream=stream
    )
    n2 = 32 * 56 * 16
    prep_scales_kernel(w2_s, rr2, by2, Int32(n2)).launch(
        grid=(64, 1, 1), block=(XFORM_THREADS, 1, 1), stream=stream
    )
    fold13_kernel(w13, rr13, w13f, Int32(32 * 4096 * 112)).launch(
        grid=(XFORM_CTAS, 1, 1), block=(XFORM_THREADS, 1, 1), stream=stream
    )
    fold2_kernel(w2, rr2, w2f, Int32(32 * 7168 * 32)).launch(
        grid=(XFORM_CTAS, 1, 1), block=(XFORM_THREADS, 1, 1), stream=stream
    )
    expand_sf_atoms_kernel(by13, sfb13, Int32(n13)).launch(
        grid=(224, 1, 1), block=(XFORM_THREADS, 1, 1), stream=stream
    )
    expand_sf_atoms_kernel(by2, sfb2, Int32(n2)).launch(
        grid=(64, 1, 1), block=(XFORM_THREADS, 1, 1), stream=stream
    )
