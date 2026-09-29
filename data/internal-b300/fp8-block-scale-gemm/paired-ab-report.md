# Cooperative fused activation-scale pack

Incumbent source commit: `403eb18`.
Candidate source commit: `f7a210d`.
Authoritative submission: `32487b38b04f461b`.

The large path converts the live FP32 activation scales in the resident CuTe
GEMM grid, publishes the packed E8M0 words through the async-global proxy, and
uses a reusable device-wide count/epoch barrier before the existing TMA
mainloop. The cooperative launch is the exact 148-CTA B300 residency cap (74
two-CTA clusters). The M64 path retains the separate Triton pack.

## Correctness

The expanded candidate/control matrix passed twice, on independent B300
leases:

- run and prepared-run equality
- M511/512/513 and M385/768/769/4097 boundaries
- alternating live-scale mutation at M64 and M4096
- 25-iteration race stress at M256/257/385/511/512/513/769/4096
- a non-default-stream M4096 run

The task fast check passed 1/1 at 45.776 us, the task full check passed 1/1 at
46.802 us, and the submission-trigger fast check passed 1/1 at 46.923 us.

## Paired uninstrumented CUPTI

The harness brackets one candidate total invocation with the committed
two-launch control (and reverses the order on alternating trials). It performs
48 M4096 and 24 M64 trials after the correctness matrix.

| B300 lease | M4096 candidate | M4096 control | median benefit | bootstrap 95% CI | wins |
|---|---:|---:|---:|---:|---:|
| first | 47.018 us | 48.461 us | 1.468 us | [1.3895, 1.5263] us | 48/48 |
| second | 46.989 us | 47.688 us | 0.7455 us | [0.6760, 0.7957] us | 48/48 |

M64 remained neutral:

| B300 lease | candidate | control | median benefit | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| first | 13.8079 us | 13.8161 us | 0.0036 us | [-0.0875, 0.1443] us |
| second | 13.4768 us | 13.5716 us | 0.0440 us | [-0.0005, 0.1273] us |

## Authoritative result

The out-of-docker judge accepted correctness 1/1 with large/all geomean
`1.6780x`, improving the prior accepted `1.6261x`. Small is not present in the
single production workload and is reported as `n/a`.

