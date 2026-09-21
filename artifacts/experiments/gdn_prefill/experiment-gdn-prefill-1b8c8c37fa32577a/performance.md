# Performance and correctness evidence

The quotations below are exact retained trace substrings. Source-session filesystem locators were not retained.
Analysis is labeled separately and does not claim a stronger result than the quotation.

## Comparison 1

### Exact retained excerpt

```text
\| T=5709, N=2 (value split 2) \| 168.944 us \| 165.759 us \| -1.89% \|
```

### English interpretation

The inherited-to-selected comparison reports 1.89% lower latency (1.02× faster).

Structured result: `latency` changed from `168.944 us` to `165.759 us`.

## Comparison 2

### Exact retained excerpt

```text
\| NCU duration \| 14.18 us \| 13.76 us \| -2.96% \|
```

### English interpretation

The inherited-to-selected comparison reports 2.96% lower latency (1.03× faster).

Structured result: `latency` changed from `14.18 us` to `13.76 us`.

## Comparison 3

### Exact retained excerpt

```text
\| 4124 / 15, split 2 \| 1.62 \| 36.976 us \| 36.701 us \| -0.75% \|
```

### English interpretation

The inherited-to-selected comparison reports 0.74% lower latency (1.01× faster).

Structured result: `latency` changed from `36.976 us` to `36.701 us`.

## Comparison 4

### Exact retained excerpt

```text
\| 8192 / 32 \| 1.73 \| 87.906 us \| 87.681 us \| -0.26% \|
```

### English interpretation

The inherited-to-selected comparison reports 0.26% lower latency (1× faster).

Structured result: `latency` changed from `87.906 us` to `87.681 us`.

## Comparison 5

### Exact retained excerpt

```text
\| 8192 / 34 \| 1.84 \| 121.314 us \| 121.137 us \| -0.15% \|
```

### English interpretation

The inherited-to-selected comparison reports 0.15% lower latency (1× faster).

Structured result: `latency` changed from `121.314 us` to `121.137 us`.

## Correctness

Status: `gate-described-no-explicit-result`.

suite and the large geomean, with every workload passing correctness.
