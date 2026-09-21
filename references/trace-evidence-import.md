# Optimization Trace Import Design

This document records the deterministic intake policy for optimization-trace
evidence. The import tree is an explicit, read-only input to
`scripts/import_trace_evidence.py`; it is never a runtime dependency of the
wiki.

## Identity and receipts

- A manifest row is identified by the SHA-256 of its canonical JSON form.
- A logical run ID is a one-way digest of the input run identity. Source
  filesystem locators are never persisted.
- Experiment IDs are derived from task family, logical run identity, the
  selected logical code path, and the newline-normalized evidence digest.
- Import receipts use `receipt_date: unknown` with the deterministic policy
  `evidence-preserved-unknown`; wall-clock import time is intentionally omitted
  so identical input produces byte-identical output.
- Source code is copied byte-for-byte after CRLF/CR-to-LF normalization. The
  provenance receipt says which normalization mode applied and records every
  payload hash.

## Acceptance policy

A `verified_faster` row is accepted only when the manifest agrees with the
evidence metadata and all of the following can be established without a
guess:

1. The task, complete before source, complete after source, and imported diff
   sections parse successfully.
2. Before and after are non-empty, different, and free of external workspace
   locators.
3. The imported diff agrees with the complete source pair after insignificant
   trailing blank context is normalized. The local `changes.diff` is then
   regenerated from the local pair.
4. At least one retained comparison contains explicit old and new values and
   has a positive direction for the stated metric. A speedup against an
   unrelated external baseline is not a before/after comparison.
5. The comparison is not a generic example, a rejected experiment, or an
   internally conflicting result.
6. The task is English canonical content and names enough architecture and
   implementation-language evidence to use controlled vocabulary safely.
7. Correctness is either explicitly reported, or the page records the exact
   task-level correctness gate and states that the retained trace does not
   contain a separate correctness result.

Rows that fail a condition are retained only in the local import ledger with a
stable rejection reason. Exact duplicate evidence/code tuples point to one
canonical accepted row. Similar rows with distinct code are not merged; their
semantic relationship is reported in the ledger.

Known unsafe performance snippets, including generic documentation examples
that happen to contain numeric improvements, are rejected rather than repaired
or reinterpreted.

## Local representation

- Source pages live at `sources/experiments/<task-family>.md` and use the
  aggregate `source-experiment` schema. Per-experiment logical IDs and hashes
  remain in the page's `experiments` records and artifact receipts.
- Payloads live under `artifacts/experiments/` and use the
  `experiment-bundle-provenance` contract, which deliberately does not invent
  an upstream URL, repository, commit, author, or license.
- One synthesized case-study index per task family lives under `wiki/kernels/`
  without a hash suffix and uses `experimental` confidence. Complete code and
  diffs remain in the linked per-experiment artifact bundles.
- Performance locators point to anchors in the local `performance.md` file.
- `data/trace-import-ledger.jsonl` is the deterministic, one-row-per-input
  disposition ledger.

The task transcript is stored as a local, English evidence summary rather than
copying operational workspace commands or stale paths. Complete before and
after code remains exact under the documented newline policy.
