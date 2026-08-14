---
id: doc-nvidia-tuning-guide
title: "NVIDIA Blackwell Tuning Guide"
url: https://docs.nvidia.com/cuda/blackwell-tuning-guide/
source_category: official-doc
architectures: [sm100, sm100a]
tags: [tcgen05, tmem, clc, tma, 2sm-cooperative, nvfp4, fp8, fp4, block-scale, pdl, gdc]
retrieved_at: 2026-08-13
---

# NVIDIA Blackwell Tuning Guide

## Scope

The Blackwell Tuning Guide describes compute capability 10.0 behavior and
porting considerations. Exact product specifications such as enabled SM count,
L2 size, and HBM capacity/bandwidth vary by product and configuration; they
should come from the relevant NVIDIA product datasheet rather than be inferred
from compute capability.

## Tensor Core and TMEM model

SM100 adds the fifth-generation Tensor Core PTX family. `tcgen05.mma` uses TMEM
for its destination/accumulator and is issued by an elected thread. Legal
instruction shapes, operand sources, layouts, data types, scale formats, and
target requirements are defined by the PTX ISA and are not captured by one
fixed “maximum shape” table.

The SM100 TMEM view documented by PTX has 512 columns by 128 lanes of 32-bit
cells. Allocation is warp-collective, uses a 32-column unit and a power-of-two
column count, and must be explicitly deallocated before kernel exit.

## Cluster Launch Control

CLC is a work-stealing mechanism. A running CTA/cluster can asynchronously try
to cancel a not-yet-launched grid item and, on success, process the returned CTA
ID. This can improve load balance or last-wave utilization, including under
partial SM availability. It does not replace the CUDA grid with a free-form
hardware tile queue, and it cannot cancel an arbitrary caller-selected tile.

## TMA and layouts

TMA performs asynchronous bulk tensor copies and can multicast shared operands
within a cluster. Completion is coordinated with mbarriers. Tensor-map swizzle
modes are optional layout choices subject to descriptor and consumer
constraints. The PTX ISA explicitly supports multiple swizzle modes for common
K-major `tcgen05.mma` operand layouts; 128-byte swizzling is not universally
required for correctness.

## Programmatic Dependent Launch

PDL is a distinct, opt-in inter-kernel launch feature. It uses a launch
attribute and device-side trigger/wait APIs. It must not be described as a
default Blackwell scheduling mode or as Cluster Launch Control.

## Performance guidance

Pipeline depth, swizzle choice, CTA grouping, persistent scheduling, and warp
specialization are workload-dependent. Tutorial measurements such as the
`tcgen05 for dummies` progression are valid only for the stated kernel, shapes,
software versions, clocks, and B200 setup; they are not hardware guarantees or
an official tuning-guide progression.

## Authoritative sources

- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Programming Guide: Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)
- [CUDA Programming Guide: Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
