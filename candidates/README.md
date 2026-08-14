# Candidate Ledgers

YAML snapshots recording include/exclude/defer decisions for each repo's PR
candidates. They are committed as a refresh audit trail. An `include` row must
resolve to a generated page or an explicit skip record; the reverse is not an
invariant because retained historical pages can predate the current snapshot or
remain after a later review changes their decision.
