# Performance and correctness evidence

The quotations below are exact retained trace substrings. Source-session filesystem locators were not retained.
Analysis is labeled separately and does not claim a stronger result than the quotation.

## Comparison 1

### Exact retained excerpt

```text
median fell from 0.2351 ms to 0.1535 ms. The scalar T=901 dispatch preserves
```

### English interpretation

Latency decreases from 0.2351 ms to 0.1535 ms: 34.71% lower, or approximately 1.53× faster.

Structured result: `latency` changed from `0.2351 ms` to `0.1535 ms`.

## Comparison 2

### Exact retained excerpt

```text
to 1.128527 ms. Primary large geomean rose from 1.6938x to 1.7380x (+2.61%);
```

### English interpretation

The reported performance metric increases from 1.6938 to 1.738, a 2.61% improvement.

Structured result: `score` changed from `1.6938 ratio` to `1.738 ratio`.

## Comparison 3

### Exact retained excerpt

```text
ms. Primary large geomean rose from 1.7380x to 1.7668x (+1.66%); all-workload
```

### English interpretation

The reported performance metric increases from 1.738 to 1.7668, a 1.66% improvement.

Structured result: `score` changed from `1.738 ratio` to `1.7668 ratio`.

## Comparison 4

### Exact retained excerpt

```text
+ms. Primary large geomean rose from 1.7380x to 1.7668x (+1.66%); all-workload
```

### English interpretation

The reported performance metric increases from 1.738 to 1.7668, a 1.66% improvement.

Structured result: `score` changed from `1.738 ratio` to `1.7668 ratio`.

## Correctness

Status: `gate-described-no-explicit-result`.

with every workload passing correctness.
