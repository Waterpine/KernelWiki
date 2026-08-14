---
id: pattern-low-sm-utilization
title: "Low SM Utilization"
type: pattern
tags: [persistent-kernel, clc, tile-scheduling]
symptoms: [low-sm-utilization, tail-effect, load-imbalance]
candidate_techniques: [technique-persistent-kernels, technique-tile-scheduling, hw-clc]
related: [pattern-tail-effect, pattern-compute-bound]
sources: [doc-ptx-isa-sm100, doc-cutlass-clc, doc-nvidia-tuning-guide, blog-tcgen05-tutorial, pr-cutlass-2161]
---

## Diagnosis

Determine whether idle capacity comes from a small grid, cluster placement,
resource-limited residency, wave quantization, or variable work duration.
“Utilization below 60%” is not an architectural threshold.

| Candidate | Scope |
|---|---|
| Larger/smaller tiles | Changes work-item count and per-item efficiency. |
| Static or software-persistent mapping | Lets a resident CTA process multiple logical tiles. |
| CLC | Can claim not-yet-launched CTA/cluster IDs; it does not assign arbitrary work. |
| Swizzle/raster change | Changes locality/order, not the number of physical workers. |

CLC cannot eliminate all imbalance, and persistence does not eliminate kernel
launch overhead (the persistent kernel still launches). Compare variants using
the same tile mapping and count cancellation/request overhead.
