# Performance and correctness evidence

The quotations below are exact retained trace substrings. Source-session filesystem locators were not retained.
Analysis is labeled separately and does not claim a stronger result than the quotation.

## Comparison 1

### Exact retained excerpt

```text
\| 1 \| tiny PAD64 \| 126.144 us \| 125.872 us \| -0.190% \| 861 / 731 / 8 \|
```

### English interpretation

The inherited-to-selected comparison reports 0.22% lower latency (1× faster).

Structured result: `latency` changed from `126.144 us` to `125.872 us`.

## Comparison 2

### Exact retained excerpt

```text
\| 7 \| tiny PAD64 \| 151.264 us \| 150.944 us \| -0.212% \| 876 / 719 / 5 \|
```

### English interpretation

The inherited-to-selected comparison reports 0.21% lower latency (1× faster).

Structured result: `latency` changed from `151.264 us` to `150.944 us`.

## Comparison 3

### Exact retained excerpt

```text
\| 52 \| tiny PAD64 \| 230.816 us \| 230.592 us \| -0.069% \| 822 / 769 / 9 \|
```

### English interpretation

The inherited-to-selected comparison reports 0.10% lower latency (1× faster).

Structured result: `latency` changed from `230.816 us` to `230.592 us`.

## Comparison 4

### Exact retained excerpt

```text
\| 80 \| tiny PAD64 \| 322.496 us \| 322.224 us \| -0.050% \| 822 / 773 / 5 \|
```

### English interpretation

The inherited-to-selected comparison reports 0.08% lower latency (1× faster).

Structured result: `latency` changed from `322.496 us` to `322.224 us`.

## Correctness

Status: `gate-described-no-explicit-result`.

with every workload passing correctness.
