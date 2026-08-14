---
id: doc-cutile-python-dsl
title: "cuTile Python Reference"
url: https://docs.nvidia.com/cuda/cutile-python/
source_category: official-doc
architectures: [sm100, sm100a, sm90]
tags: [cutile, python, gemm, tma]
retrieved_at: 2026-08-13
---

# cuTile Python

cuTile is NVIDIA's Python tile-programming DSL. Kernels run over a logical grid
of blocks, use `@ct.kernel`, and are launched from the host with `ct.launch`.
Arrays are mutable, strided global-memory objects; tiles are immutable kernel
values with compile-time, power-of-two dimensions.

```python
import cuda.tile as ct

@ct.kernel
def vector_add(a, b, out, tile_size: ct.Constant[int]):
    block = ct.bid(0)
    result = ct.load(a, (block,), (tile_size,)) + ct.load(b, (block,), (tile_size,))
    ct.store(out, (block,), result)
```

## Current requirements and versions

The quickstart accessed 2026-08-13 requires driver R580 or newer, Python
3.10--3.14/3.14t, and a GPU in compute-capability family 8.x, 9.x, 10.x, 11.x,
or 12.x. It can use a system CUDA Toolkit 13.1+ or install matching TileIR,
NVCC, and NVVM packages through `cuda-tile[tileiras]`.

The current cuTile Python release is 1.5.0 (2026-07-08). Release 1.4.0 with
CUDA 13.3 added Hopper support and `ct.mma_scaled()`; therefore the former
SM100-only architecture list and fixed “introduced in CUDA 13.1” description
were incomplete.

The compiler selects lower-level mechanisms for the target. Source-level
portability does not guarantee that every load uses TMA, every matmul uses a
particular tcgen05 form, or every accumulator resides in TMEM. Those are
lowering and target decisions that must be confirmed from generated code and
profiling.

## Sources

- [Documentation and execution model](https://docs.nvidia.com/cuda/cutile-python/)
- [Quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html)
- [Release notes](https://docs.nvidia.com/cuda/cutile-python/generated/release_notes.html)
