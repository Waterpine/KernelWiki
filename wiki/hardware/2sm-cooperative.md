---
id: hw-2sm-cooperative
title: "Two-CTA Cooperative tcgen05 Operations"
type: hardware
architectures: [sm100, sm100a]
tags: [2sm-cooperative, tcgen05, cluster]
confidence: verified
related: [hw-tcgen05-mma, hw-tmem, technique-warp-specialization]
sources: [doc-ptx-isa-sm100, doc-cutlass-blackwell, pr-cutlass-2139, pr-cutlass-3106, blog-colfax-cutlass, blog-modular-blackwell, blog-tcgen05-tutorial]
aliases: ["2-SM cooperative", "dual CTA", "2CTA", "cta_group::2"]
evidence_basis:
  - source_id: doc-ptx-isa-sm100
    evidence_type: official-doc
  - source_id: pr-cutlass-2139
    evidence_type: upstream-code
---

## Overview

The PTX qualifier `cta_group::2` makes a pair of peer CTAs cooperate in the
tcgen05 operation. The peer relationship and launch placement are defined by
the cluster and CTA-pair rules. It should not be described as a generic promise
that “two SMs in one TPC” always execute one fixed `m256n256k16` tile.

All tcgen05 instructions in a kernel must use the same CTA-group value. For
TMEM allocation/deallocation in group-2 mode, one warp from each peer CTA
participates collectively, and the issuing warp must ensure the peer CTA has
launched and will reach its collective operations.

```ptx
// Schematic MMA family; exact idesc and predicate depend on the selected form.
.reg .b32 d;
.reg .b64 a_desc, b_desc;
.reg .pred p;
tcgen05.mma.cta_group::2.kind::f16
    [d], a_desc, b_desc, idesc, p;
```

## Shape and memory model

Group-2 changes the CTA participation and distributed TMEM contract. Legal
M/N/K shapes still come from the chosen MMA kind, types, layouts, and
instruction descriptor. Each peer has its own shared-memory and TMEM view; the
implementation must follow the PTX peer-CTA addressing, multicast, allocation,
completion, and deallocation rules.

CUTLASS/CuTe exposes `CtaGroup.TWO` atoms and matching copy/pipeline helpers.
Those versioned abstractions are safer than copying a group-1 descriptor and
changing only the qualifier.

## Performance scope

Two-CTA operation can enable a different tile decomposition or data-reuse
strategy, but it is not inherently faster. The tutorial's reported movement
from 80% to 86% of a library baseline is one multi-step implementation result,
not a 7.5% architectural guarantee or a rule that all problems with M >= 256
should use group-2.
