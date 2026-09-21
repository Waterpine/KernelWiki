# Performance and correctness evidence

The quotations below are exact retained trace substrings. Source-session filesystem locators were not retained.
Analysis is labeled separately and does not claim a stronger result than the quotation.

## Comparison 1

### Exact retained excerpt

```text
recurrence fell from 4.436 ms to 4.224 ms in the launch profiler, and the fast
```

### English interpretation

Latency decreases from 4.436 ms to 4.224 ms: 4.78% lower, or approximately 1.05× faster.

Structured result: `latency` changed from `4.436 ms` to `4.224 ms`.

## Comparison 2

### Exact retained excerpt

```text
official H96 timing fell from 5.589 ms to 5.343 ms; H64 remained at 3.047 ms.
```

### English interpretation

Latency decreases from 5.589 ms to 5.343 ms: 4.40% lower, or approximately 1.05× faster.

Structured result: `latency` changed from `5.589 ms` to `5.343 ms`.

## Comparison 3

### Exact retained excerpt

```text
The fast geomean rose from 0.2381x to 0.2435x.
```

### English interpretation

The reported performance metric increases from 0.2381 to 0.2435, a 2.27% improvement.

Structured result: `score` changed from `0.2381 ratio` to `0.2435 ratio`.

## Comparison 4

### Exact retained excerpt

```text
launch breakdown shows preparation falling from 1.167 ms to 1.124 ms on H96
```

### English interpretation

Latency decreases from 1.167 ms to 1.124 ms: 3.68% lower, or approximately 1.04× faster.

Structured result: `latency` changed from `1.167 ms` to `1.124 ms`.

## Comparison 5

### Exact retained excerpt

```text
fixed and from 0.786 ms to 0.749 ms on H64 fixed. Two fast runs were stable at
```

### English interpretation

Latency decreases from 0.786 ms to 0.749 ms: 4.71% lower, or approximately 1.05× faster.

Structured result: `latency` changed from `0.786 ms` to `0.749 ms`.

## Comparison 6

### Exact retained excerpt

```text
instrumented solve/staging phase from 2.251 us to 1.670 us and total first-tile
```

### English interpretation

Latency decreases from 2.251 us to 1.670 us: 25.81% lower, or approximately 1.35× faster.

Structured result: `latency` changed from `2.251 us` to `1.67 us`.

## Comparison 7

### Exact retained excerpt

```text
time from 7.226 us to 6.536 us.
```

### English interpretation

Latency decreases from 7.226 us to 6.536 us: 9.55% lower, or approximately 1.11× faster.

Structured result: `latency` changed from `7.226 us` to `6.536 us`.

## Comparison 8

### Exact retained excerpt

```text
again from 5.10 ms to 4.60 ms, while H64 remains on its unchanged M64 warp
```

### English interpretation

Latency decreases from 5.10 ms to 4.60 ms: 9.80% lower, or approximately 1.11× faster.

Structured result: `latency` changed from `5.1 ms` to `4.6 ms`.

## Comparison 9

### Exact retained excerpt

```text
1.018 us to 0.552 us and total first-tile time from 6.536 us to 5.962 us.
```

### English interpretation

Latency decreases from 6.536 us to 5.962 us: 8.78% lower, or approximately 1.1× faster.

Structured result: `latency` changed from `6.536 us` to `5.962 us`.

## Comparison 10

### Exact retained excerpt

```text
1.848 us and total first-tile time from 5.962 us to 5.130 us.
```

### English interpretation

Latency decreases from 5.962 us to 5.130 us: 13.96% lower, or approximately 1.16× faster.

Structured result: `latency` changed from `5.962 us` to `5.13 us`.

## Comparison 11

### Exact retained excerpt

```text
0.970 us and total recurrence time falls from 3.207 us to 2.641 us. Racecheck
```

### English interpretation

Latency decreases from 3.207 us to 2.641 us: 17.65% lower, or approximately 1.21× faster.

Structured result: `latency` changed from `3.207 us` to `2.641 us`.

## Comparison 12

### Exact retained excerpt

```text
comparison improves H64 fixed from 2.692 ms to 2.661 ms with exact matching;
```

### English interpretation

Latency decreases from 2.692 ms to 2.661 ms: 1.15% lower, or approximately 1.01× faster.

Structured result: `latency` changed from `2.692 ms` to `2.661 ms`.

## Comparison 13

### Exact retained excerpt

```text
1.174 us and total recurrence time falls from 5.130 us to 4.371 us. A forced
```

### English interpretation

Latency decreases from 5.130 us to 4.371 us: 14.80% lower, or approximately 1.17× faster.

Structured result: `latency` changed from `5.13 us` to `4.371 us`.

## Comparison 14

### Exact retained excerpt

```text
falling from 1.538 us to 1.513 us and total recurrence from 4.371 us to
```

### English interpretation

Latency decreases from 1.538 us to 1.513 us: 1.63% lower, or approximately 1.02× faster.

Structured result: `latency` changed from `1.538 us` to `1.513 us`.

## Comparison 15

### Exact retained excerpt

```text
falling from 1.513 us to 1.441 us and total recurrence from 4.358 us to
```

### English interpretation

Latency decreases from 1.513 us to 1.441 us: 4.76% lower, or approximately 1.05× faster.

Structured result: `latency` changed from `1.513 us` to `1.441 us`.

## Comparison 16

### Exact retained excerpt

```text
0.925 us and total recurrence from 4.275 us to 3.441 us. H96 fixed improves
```

### English interpretation

Latency decreases from 4.275 us to 3.441 us: 19.51% lower, or approximately 1.24× faster.

Structured result: `latency` changed from `4.275 us` to `3.441 us`.

## Comparison 17

### Exact retained excerpt

```text
staging falling from 1.024 us to 0.843 us and total recurrence from 3.053 us
```

### English interpretation

Latency decreases from 1.024 us to 0.843 us: 17.68% lower, or approximately 1.21× faster.

Structured result: `latency` changed from `1.024 us` to `0.843 us`.

## Comparison 18

### Exact retained excerpt

```text
On the official fast workloads, H96 improves from 1.176613 ms to 0.975398 ms
```

### English interpretation

Latency decreases from 1.176613 ms to 0.975398 ms: 17.10% lower, or approximately 1.21× faster.

Structured result: `latency` changed from `1.176613 ms` to `0.975398 ms`.

## Comparison 19

### Exact retained excerpt

```text
and H64 from 0.924707 ms to 0.739955 ms.  Both pass with 100% element matching;
```

### English interpretation

Latency decreases from 0.924707 ms to 0.739955 ms: 19.98% lower, or approximately 1.25× faster.

Structured result: `latency` changed from `0.924707 ms` to `0.739955 ms`.

## Correctness

Status: `gate-described-no-explicit-result`.

benchmark workload set, with every workload passing correctness.
