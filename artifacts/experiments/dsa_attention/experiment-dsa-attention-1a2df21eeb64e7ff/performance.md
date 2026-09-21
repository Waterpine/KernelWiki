# Performance and correctness evidence

The quotations below are exact retained trace substrings. Source-session filesystem locators were not retained.
Analysis is labeled separately and does not claim a stronger result than the quotation.

## Comparison 1

### Exact retained excerpt

```text
wl20 from 11.33us to 10.91us.  Hoisting the address base uses 248 registers
```

### English interpretation

Latency decreases from 11.33 us to 10.91 us: 3.71% lower, or approximately 1.04× faster.

Structured result: `latency` changed from `11.33 us` to `10.91 us`.

## Correctness

Status: `gate-described-no-explicit-result`.

Any failing workload zeroes the official score, so correctness on
