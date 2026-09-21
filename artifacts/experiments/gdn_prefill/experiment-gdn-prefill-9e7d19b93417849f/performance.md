# Performance and correctness evidence

The quotations below are exact retained trace substrings. Source-session filesystem locators were not retained.
Analysis is labeled separately and does not claim a stronger result than the quotation.

## Comparison 1

### Exact retained excerpt

```text
\| instrumented K1 \| 109.952 us \| 101.920 us \| -7.31% \|
```

### English interpretation

The inherited-to-selected comparison reports 7.31% lower latency (1.08× faster).

Structured result: `latency` changed from `109.952 us` to `101.92 us`.

## Comparison 2

### Exact retained excerpt

```text
\| mean K1 CTA lifetime \| 18.470 us \| 16.939 us \| -8.29% \|
```

### English interpretation

The inherited-to-selected comparison reports 8.29% lower latency (1.09× faster).

Structured result: `latency` changed from `18.47 us` to `16.939 us`.

## Comparison 3

### Exact retained excerpt

```text
\| K1 CTA start spread \| 91.616 us \| 84.992 us \| -7.23% \|
```

### English interpretation

The inherited-to-selected comparison reports 7.23% lower latency (1.08× faster).

Structured result: `latency` changed from `91.616 us` to `84.992 us`.

## Comparison 4

### Exact retained excerpt

```text
\| instrumented K23 \| 58.528 us \| 56.768 us \| -3.01% \|
```

### English interpretation

The inherited-to-selected comparison reports 3.01% lower latency (1.03× faster).

Structured result: `latency` changed from `58.528 us` to `56.768 us`.

## Comparison 5

### Exact retained excerpt

```text
\| mean K23 CTA lifetime \| 17.355 us \| 15.863 us \| -8.59% \|
```

### English interpretation

The inherited-to-selected comparison reports 8.60% lower latency (1.09× faster).

Structured result: `latency` changed from `17.355 us` to `15.863 us`.

## Comparison 6

### Exact retained excerpt

```text
\| K23 CTA start spread \| 27.936 us \| 25.312 us \| -9.39% \|
```

### English interpretation

The inherited-to-selected comparison reports 9.39% lower latency (1.1× faster).

Structured result: `latency` changed from `27.936 us` to `25.312 us`.

## Comparison 7

### Exact retained excerpt

```text
\| `UWcopy` mean \| 1.1365 us \| 0.7710 us \| -32.16% \|
```

### English interpretation

The inherited-to-selected comparison reports 32.16% lower latency (1.47× faster).

Structured result: `latency` changed from `1.1365 us` to `0.771 us`.

## Comparison 8

### Exact retained excerpt

```text
\| `Ks` + `UWcopy` mean \| 2.1769 us \| 2.1190 us \| -2.66% \|
```

### English interpretation

The inherited-to-selected comparison reports 2.66% lower latency (1.03× faster).

Structured result: `latency` changed from `2.1769 us` to `2.119 us`.

## Comparison 9

### Exact retained excerpt

```text
\| `Mbuild` + `att` mean \| 3.5465 us \| 3.4707 us \| -2.14% \|
```

### English interpretation

The inherited-to-selected comparison reports 2.14% lower latency (1.02× faster).

Structured result: `latency` changed from `3.5465 us` to `3.4707 us`.

## Comparison 10

### Exact retained excerpt

```text
\| `Mbuild` p50 \| 3.008 us \| 2.944 us \| -2.13% \|
```

### English interpretation

The inherited-to-selected comparison reports 2.13% lower latency (1.02× faster).

Structured result: `latency` changed from `3.008 us` to `2.944 us`.

## Comparison 11

### Exact retained excerpt

```text
\| instrumented K1 \| 111.616 us \| 109.184 us \| -2.18% \|
```

### English interpretation

The inherited-to-selected comparison reports 2.18% lower latency (1.02× faster).

Structured result: `latency` changed from `111.616 us` to `109.184 us`.

## Comparison 12

### Exact retained excerpt

```text
\| `Mbuild` + `att` mean \| 3.5465 us \| 3.4707 us \| -2.14% \|
```

### English interpretation

The inherited-to-selected comparison reports 2.14% lower latency (1.02× faster).

Structured result: `latency` changed from `3.5465 us` to `3.4707 us`.

## Comparison 13

### Exact retained excerpt

```text
\| K1 `epi` mean \| 2.5045 us \| 2.4718 us \| -1.31% \|
```

### English interpretation

The inherited-to-selected comparison reports 1.31% lower latency (1.01× faster).

Structured result: `latency` changed from `2.5045 us` to `2.4718 us`.

## Correctness

Status: `gate-described-no-explicit-result`.

with every workload passing correctness.
