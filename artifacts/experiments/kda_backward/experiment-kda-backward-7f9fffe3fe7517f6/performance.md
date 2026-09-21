# Performance and correctness evidence

The quotations below are exact retained trace substrings. Source-session filesystem locators were not retained.
Analysis is labeled separately and does not claim a stronger result than the quotation.

## Comparison 1

### Exact retained excerpt

```text
The complete-body skew falls from 1.632 us to 0.320 us, and the epilogue skew
```

### English interpretation

Latency decreases from 1.632 us to 0.320 us: 80.39% lower, or approximately 5.1× faster.

Structured result: `latency` changed from `1.632 us` to `0.32 us`.

## Comparison 2

### Exact retained excerpt

```text
falls from 1.664 us to 0.288 us. The moved work does not expose either tensor
```

### English interpretation

Latency decreases from 1.664 us to 0.288 us: 82.69% lower, or approximately 5.78× faster.

Structured result: `latency` changed from `1.664 us` to `0.288 us`.

## Correctness

Status: `gate-described-no-explicit-result`.

FULL benchmark workload set, with every workload passing correctness.
