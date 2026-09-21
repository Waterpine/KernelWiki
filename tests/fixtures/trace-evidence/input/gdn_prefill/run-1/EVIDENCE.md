# Verified Optimization Evidence: Optimize fixture GDN prefill on NVIDIA B300 (CuTe-DSL)

> **Result: the optimized code is measurably faster than the reconstructed pre-optimization code.**

## Metadata

| Field | Value |
|---|---|
| Task family | `gdn_prefill` |
| Run | `gdn_prefill/run-1` |
| Compared source file | `solution/kernel.py` |
| Trace clients | fixture: 1 |
| Trace files | 1 |
| Trace bytes | 100 |
| Before-code evidence | logical snapshot A |
| After-code evidence | logical snapshot B |

## Task

<details>
<summary>Original TASK.md</summary>

```markdown
# Optimize fixture GDN prefill on NVIDIA B300 (CuTe-DSL)

Optimize the kernel while every workload continues to pass the correctness check.
```

</details>

## Why the optimized version is faster

- **Measured improvement:** Latency decreases from 10.0 us to 8.0 us: 20.0% lower, or 1.25x faster.
- **Code change:** The optimized version removes an identity addition.

> The measured statement is observed; the mechanism is an inference.

## Complete code before optimization

```python
def run(x):
    return x + 0
```

## Complete code after optimization

```python
def run(x):
    return x
```

## Complete unified diff

```diff
--- before/solution/kernel.py
+++ after/solution/kernel.py
@@ -1,2 +1,2 @@
 def run(x):
-    return x + 0
+    return x
```

## Verified performance evidence

### Latency Reduction

- Interpretation: Latency decreases from 10.0 us to 8.0 us: 20.0% lower, or approximately 1.25x faster.
- Source: logical comparison 1
- Trace text: paired median changed from 10.0 us to 8.0 us with every workload passing correctness.

## Evidence rules

- Complete code and a positive comparison are required.
