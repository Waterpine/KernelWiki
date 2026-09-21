# Performance and correctness evidence

The quotations below are exact retained trace substrings. Source-session filesystem locators were not retained.
Analysis is labeled separately and does not claim a stronger result than the quotation.

## Comparison 1

### Exact retained excerpt

```text
\| NCU duration \| 296.352 us \| 270.464 us \| -8.74% \|
```

### English interpretation

The inherited-to-selected comparison reports 8.74% lower latency (1.1× faster).

Structured result: `latency` changed from `296.352 us` to `270.464 us`.

## Comparison 2

### Exact retained excerpt

```text
\| NCU duration \| 270.464 us \| 261.728 us \| -3.23% \|
```

### English interpretation

The inherited-to-selected comparison reports 3.23% lower latency (1.03× faster).

Structured result: `latency` changed from `270.464 us` to `261.728 us`.

## Comparison 3

### Exact retained excerpt

```text
\| `9a5d694b` \| T8192/N57 \| 306.437 us \| 287.572 us \| -6.16% \|
```

### English interpretation

The inherited-to-selected comparison reports 6.16% lower latency (1.07× faster).

Structured result: `latency` changed from `306.437 us` to `287.572 us`.

## Correctness

Status: `gate-described-no-explicit-result`.

with every workload passing correctness.
