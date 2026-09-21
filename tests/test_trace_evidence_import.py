import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "trace-evidence" / "input"


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


importer = load_script("trace_importer", "import_trace_evidence.py")
self_check = load_script("trace_self_check", "check_self_contained.py")
validator = load_script("trace_validator", "validate.py")


class TraceEvidenceImportTests(unittest.TestCase):
    def test_fixture_is_accepted_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            first, first_ledger, first_summary = importer.build_import(FIXTURE, repo)
            second, second_ledger, second_summary = importer.build_import(FIXTURE, repo)
        self.assertEqual(first, second)
        self.assertEqual(first_ledger, second_ledger)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual({"accepted": 1}, first_summary["dispositions"])
        accepted = first_ledger[0]
        self.assertEqual("experiment-family-gdn-prefill", accepted["source_id"])
        self.assertEqual("kernel-trace-gdn-prefill", accepted["wiki_id"])
        self.assertTrue(accepted["experiment_id"].startswith("experiment-gdn-prefill-"))
        self.assertIn("sources/experiments/gdn_prefill.md", first)
        self.assertIn("wiki/kernels/kernel-trace-gdn-prefill.md", first)
        persisted = b"\n".join(first.values()).decode("utf-8")
        self.assertNotIn(str(FIXTURE), persisted)
        self.assertNotIn("gdn_prefill/run-1/EVIDENCE.md", persisted)
        self.assertNotIn("solution/kernel.py", persisted)
        provenance = next(
            content for path, content in first.items() if path.endswith("/PROVENANCE.yaml")
        ).decode("utf-8")
        self.assertIn("mode: copied-verbatim", provenance)

    def test_exact_duplicate_gets_canonical_target(self):
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as repo_dir:
            fixture_copy = Path(input_dir)
            (fixture_copy / "gdn_prefill" / "run-1").mkdir(parents=True)
            evidence = FIXTURE / "gdn_prefill" / "run-1" / "EVIDENCE.md"
            (fixture_copy / "gdn_prefill" / "run-1" / "EVIDENCE.md").write_bytes(evidence.read_bytes())
            line = (FIXTURE / "index.jsonl").read_text(encoding="utf-8").strip()
            (fixture_copy / "index.jsonl").write_text(f"{line}\n{line}\n", encoding="utf-8")
            _files, ledger, summary = importer.build_import(fixture_copy, Path(repo_dir))
        self.assertEqual({"accepted": 1, "duplicate": 1}, summary["dispositions"])
        duplicate = next(row for row in ledger if row["disposition"] == "duplicate")
        accepted = next(row for row in ledger if row["disposition"] == "accepted")
        self.assertEqual(accepted["input_id"], duplicate["canonical_input_id"])
        self.assertEqual(accepted["experiment_id"], duplicate["canonical_experiment_id"])

    def test_external_baseline_only_claim_is_rejected(self):
        row = json.loads((FIXTURE / "index.jsonl").read_text(encoding="utf-8"))
        evidence = (FIXTURE / row["evidence_path"]).read_text(encoding="utf-8")
        evidence = evidence.replace(
            "Latency decreases from 10.0 us to 8.0 us: 20.0% lower, or approximately 1.25x faster.",
            "The trace explicitly reports a 1.25x speedup against an external baseline.",
        ).replace(
            "paired median changed from 10.0 us to 8.0 us with every workload passing correctness.",
            "The selected kernel is 1.25x faster than the external baseline.",
        )
        parsed, reason = importer.parse_evidence(row, evidence.encode("utf-8"))
        self.assertIsNone(parsed)
        self.assertEqual("no_direct_before_after_comparison", reason)

    def test_written_fixture_is_self_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            files, _ledger, _summary = importer.build_import(FIXTURE, repo)
            importer.write_atomic(repo, files, force=False)
            errors = self_check.check_repository(repo, ["definitely-not-present-source-token"])
        self.assertEqual([], errors)

    def test_source_experiment_schema_rejects_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            page = repo / "sources" / "experiments" / "gdn_prefill" / "bad.md"
            page.parent.mkdir(parents=True)
            page.write_text("---\nid: experiment-bad\ntitle: Bad\n---\n", encoding="utf-8")
            previous = validator.REPO_ROOT
            validator.REPO_ROOT = repo
            try:
                schemas = validator.load_yaml_file(ROOT / "data" / "schemas.yaml")
                tags = validator.load_yaml_file(ROOT / "data" / "tags.yaml")
                errors = validator.validate_file(page, schemas, tags, set(), validator._load_code_langs())
            finally:
                validator.REPO_ROOT = previous
        self.assertTrue(any("missing required field 'task_family'" in error for error in errors), errors)

    def test_experiment_provenance_detects_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            files, _ledger, _summary = importer.build_import(FIXTURE, repo)
            importer.write_atomic(repo, files, force=False)
            bundle = next((repo / "artifacts" / "experiments").glob("*/*"))
            (bundle / "before.py").write_text("def changed():\n    pass\n", encoding="utf-8")
            previous = validator.REPO_ROOT
            validator.REPO_ROOT = repo
            try:
                errors = validator.validate_experiment_bundle(bundle)
            finally:
                validator.REPO_ROOT = previous
        self.assertTrue(any("sha256 mismatch" in error for error in errors), errors)

    def test_local_performance_locator_must_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            files, _ledger, _summary = importer.build_import(FIXTURE, repo)
            importer.write_atomic(repo, files, force=False)
            source = next((repo / "sources" / "experiments").rglob("*.md"))
            source.write_text(
                source.read_text(encoding="utf-8").replace("performance.md#comparison-1", "missing.md#comparison-1"),
                encoding="utf-8",
            )
            errors = self_check.page_and_path_errors(repo)
        self.assertTrue(any("source_locator" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
