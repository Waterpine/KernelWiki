# Audited MLSys26 kernel snapshots (B300)

The [complete source bundle](../../artifacts/kernels/kda-mlsys26-final/full/PROVENANCE.yaml)
copies the three kernels selected on the archived `main` of
`kernel-design-agents/mlsys-contest-cute` at commit
`1d6d4114a2717bd2d588cfaf9a5f17cf0aa8d93f` (2026-08-18). Each source
file is byte-for-byte from that commit; `PROVENANCE.yaml` records its path and
SHA-256. The entry points and their sibling modules retain the original
`kernels/` layout:

| Task | Entry point | B300 mean speedup | Correctness |
| --- | --- | ---: | ---: |
| GDN prefill | [kernel.py](../../artifacts/kernels/kda-mlsys26-final/full/kernels/gdn_prefill/kernel.py) | 15.1598× | 100/100 |
| DSA sparse attention | [kernel.py](../../artifacts/kernels/kda-mlsys26-final/full/kernels/dsa_attention/kernel.py) | 37.6827× | 23/23 |
| FP8 block-scale MoE | [kernel.py](../../artifacts/kernels/kda-mlsys26-final/full/kernels/moe/kernel.py) | 2.2227× | 19/19 |

These are arithmetic means of per-workload speedups, with any failing workload
scoring zero. The recorded full-suite acceptance runs used NVIDIA B300 with
CUPTI cold-L2 timing (10 warmups, 50 timed iterations, three trials). DSA and
MoE timed a live baseline in the same process; GDN used an attested frozen
B300 baseline table. The scores above are retained from the archived
acceptance report; row-level benchmark records are not included in this
kernel bundle. Development used Python 3.12, CUDA 13.x, PyTorch
2.12.1+cu130 and CuTe DSL 4.6.0. These B300 results are separate from the
contest's B200 leaderboard.

The recorded 2026-08-17/18 integrity audit excluded some faster raw candidates:
DSA's 45.84× candidate cached input-derived data across timed calls; MoE's
3.68× candidate changed weight precision and its lineage could return stale
output on changed inputs. A later MoE 2.3619× version was retired for
coarsening provided scale grids. Two higher-scoring GDN dev candidates failed
two of the 100 official workloads (T=35 and T=48). The bundled kernels are the
selected passing versions. The earlier A/B/C sections in this wiki describe
selected optimization steps and are not these final snapshots.

All documentation links and source-file paths above resolve inside this
KernelWiki checkout. The repository identifier and commit SHA are provenance
labels; reading the bundled code needs no external checkout. The
archived source tree has no top-level license file, while the GDN core retains
its embedded NVIDIA BSD-3-Clause notice. This local bundle has not been
rebenchmarked on the current machine.
