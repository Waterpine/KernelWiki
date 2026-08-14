---
id: technique-software-exp
title: "Software-Emulated Exponential"
type: technique
architectures: [sm100]
tags: [software-exp, attention]
confidence: verified
reproducibility: snippet
prerequisites: []
related: [kernel-flash-attention-4, technique-warp-specialization]
sources: [doc-flash-attention-4, blog-flash-attention-4, pr-flash-attention-2441]
evidence_basis:
  - source_id: doc-flash-attention-4
    evidence_type: official-doc
  - source_id: pr-flash-attention-2441
    evidence_type: upstream-code
---

## FlashAttention-4 technique

FlashAttention-4 approximates `2^x` for a tuned subset of softmax entries using
range reduction, bit construction for the integer exponent, and a polynomial
for the fractional part. It keeps the hardware exponential for the remaining
entries.

```python
def polynomial_exp2_fraction(frac, coefficients):
    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = value * frac + coefficient
    return value
```

The paper clamps the domain, computes a floor-like integer part, evaluates a
Sollya-derived polynomial on the fractional part, and recombines the result.
Its exact coefficients and rounding operations are part of the implementation;
generic Taylor coefficients are not a substitute.

The benefit is a resource-balancing tradeoff. Polynomial evaluation consumes
FMA/ALU instructions, registers, and register bandwidth. The paper therefore
emulates only roughly 10--25%, not all exponentials, and validates accuracy
after BF16 rounding. The old claims of bypassing the exponential unit entirely
and obtaining an 8x or 4x universal throughput increase were incorrect.
