---
id: pattern-register-pressure
title: "Register Pressure — Low Occupancy"
type: pattern
tags: [tmem, register-reuse, warp-specialization]
symptoms: [register-pressure, low-occupancy, register-spilling]
candidate_techniques: [hw-tmem, technique-warp-specialization, migration-register-to-tmem]
related: [pattern-compute-bound, hw-tmem]
sources: [doc-ptx-isa-sm100, doc-nvidia-tuning-guide, blog-tcgen05-tutorial, pr-vllm-16032]
---

## Diagnosis

Use compiler resource output and profiler counters to separate register-limited
residency from spills and long live ranges. Desired occupancy depends on the
kernel; lower occupancy can still be sufficient when each warp exposes enough
independent work.

TMEM moves supported tcgen05 destinations out of distributed accumulator
registers, but the epilogue must load fragments into registers and other state
remains. It does not free a fixed 128 registers per thread or guarantee another
resident CTA.

Possible changes include shortening live ranges, splitting roles, reducing
unrolling, moving eligible accumulator storage to TMEM, or accepting lower
occupancy to avoid spills. Validate both correctness and runtime after each
change.
