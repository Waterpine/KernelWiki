# Saved B300 benchmark records

The four kernel pages link directly to these recorded A/B/C measurements. The
JSONL files retain the full saved result rows, including workload axes,
`baseline_ms`, `kernel_ms`, `speedup`, and `passed`. One line is one saved suite;
the MoE full files each have a 19-row official suite and a one-row live suite.

The DSA and GDN page summaries use the arithmetic mean of passing row
speedups. The MoE and KDA backward summaries use their recorded geometric
mean. Compare candidate `kernel_ms` on matching workload IDs when baselines
change between runs. The GEMM paired report records its CUPTI A/B trial
medians and correctness checks separately.
