import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate = load_script("trace_validate", "validate.py")
indices = load_script("trace_indices", "generate-indices.py")


class TraceQueryIntegrationTests(unittest.TestCase):
    def test_experiment_page_type_is_discovered(self):
        path = ROOT / "sources" / "experiments" / "gdn_prefill.md"
        self.assertEqual(
            "source-experiment",
            validate.detect_page_type(path, {"id": "experiment-example"}),
        )

    def test_task_family_index_has_exact_import_membership(self):
        pages = indices.collect_all_pages()
        expected = sorted(page["_path"] for page in pages if page.get("task_family") == "gdn_prefill")
        rendered = indices.generate_by_task_family(pages)
        actual = sorted(page["_path"] for page in pages if f"../{page['_path']}" in rendered and page.get("task_family") == "gdn_prefill")
        self.assertEqual(expected, actual)
        self.assertEqual(
            [
                "sources/experiments/gdn_prefill.md",
                "wiki/kernels/kernel-trace-gdn-prefill.md",
            ],
            expected,
        )

    def test_task_family_filter_is_alias_aware(self):
        result = subprocess.run(
            [sys.executable, "scripts/query.py", "--task-family", "GDN prefill", "--type", "kernel", "--compact", "--limit", "100"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("kernel-trace-gdn-prefill:", result.stdout)
        self.assertNotIn("No matching pages", result.stdout)


if __name__ == "__main__":
    unittest.main()
