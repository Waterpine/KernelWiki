#!/usr/bin/env python3
"""Verify KernelWiki has no dependency on an optimization-trace input tree."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _yaml_compat import yaml  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
GENERATED_LOCATOR_PATTERNS = {
    "session locator": re.compile(r"sessions-(?:codex|claude)/", re.IGNORECASE),
    "external workspace locator": re.compile(r"/workspace/", re.IGNORECASE),
    "stale run path": re.compile(
        r"(?:^|[\s`\"'])(?:repo|solution|profiles?|bench_results)/[^\s`\"')]+",
        re.IGNORECASE | re.MULTILINE,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_repo_paths(root: Path):
    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        names[:] = [name for name in names if name not in EXCLUDED_PARTS]
        for name in names + files:
            yield directory_path / name


def text_files(root: Path):
    for path in iter_repo_paths(root):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            yield path, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def markdown_anchor(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value).strip().lower()
    value = re.sub(r"[^\w\- ]", "", value, flags=re.UNICODE)
    return value.replace(" ", "-")


def local_link_errors(root: Path) -> list[str]:
    errors = []
    link_re = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\n]+)\)")
    for path, text in text_files(root):
        if path.suffix.lower() != ".md":
            continue
        # Examples inside fenced code are not repository links.
        visible = re.sub(r"^```.*?^```\s*$", "", text, flags=re.MULTILINE | re.DOTALL)
        for match in link_re.finditer(visible):
            target = match.group(1).strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE):
                continue
            target = unquote(target)
            path_part, _, fragment = target.partition("#")
            destination = path if not path_part else path.parent / path_part
            if not destination.exists():
                errors.append(f"{path.relative_to(root)}: local link target '{target}' does not exist")
                continue
            if fragment and destination.is_file() and destination.suffix.lower() == ".md":
                try:
                    body = destination.read_text(encoding="utf-8")
                except OSError:
                    continue
                anchors = {markdown_anchor(value) for value in re.findall(r"^#{1,6}\s+(.+)$", body, re.MULTILINE)}
                anchors.update(re.findall(r"<a\s+id=[\"']([^\"']+)[\"']", body, re.IGNORECASE))
                if fragment not in anchors:
                    errors.append(
                        f"{path.relative_to(root)}: local link fragment '#{fragment}' "
                        f"does not exist in '{destination.relative_to(root)}'"
                    )
    return errors


def load_frontmatter(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return None
    try:
        value = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    return value if isinstance(value, dict) else None


def page_and_path_errors(root: Path) -> list[str]:
    errors = []
    source_ids = {}
    pages = []
    for base_name in ("sources", "wiki"):
        base = root / base_name
        if not base.is_dir():
            continue
        for path in base.rglob("*.md"):
            fm = load_frontmatter(path)
            if not fm:
                continue
            pages.append((path, fm))
            if base_name == "sources" and fm.get("id"):
                source_ids[fm["id"]] = path
    for path, fm in pages:
        rel = path.relative_to(root)
        artifact_dir = fm.get("artifact_dir")
        if artifact_dir:
            if not isinstance(artifact_dir, str) or not (root / artifact_dir).is_dir():
                errors.append(f"{rel}: artifact_dir does not resolve locally")
        for source_id in fm.get("sources") or []:
            if source_id not in source_ids:
                errors.append(f"{rel}: source_id '{source_id}' does not resolve locally")
        for claim in fm.get("performance_claims") or []:
            if not isinstance(claim, dict):
                continue
            source_id = claim.get("source_id")
            if source_id and source_id not in source_ids:
                errors.append(f"{rel}: performance source_id '{source_id}' does not resolve locally")
            locator = claim.get("source_locator")
            if fm.get("task_family") and isinstance(locator, str):
                local_path = locator.partition("#")[0]
                if not local_path.startswith("artifacts/experiments/") or not (root / local_path).is_file():
                    errors.append(f"{rel}: imported source_locator '{locator}' is not local")
    return errors


def experiment_errors(root: Path) -> list[str]:
    errors = []
    experiment_root = root / "artifacts" / "experiments"
    if not experiment_root.is_dir():
        return errors
    for provenance_path in sorted(experiment_root.rglob("PROVENANCE.yaml")):
        bundle = provenance_path.parent
        rel = bundle.relative_to(root)
        try:
            provenance = yaml.safe_load(provenance_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"{rel}: invalid provenance ({exc})")
            continue
        if not isinstance(provenance, dict):
            errors.append(f"{rel}: invalid provenance mapping")
            continue
        declared = set()
        role_paths = {}
        for entry in provenance.get("files") or []:
            if not isinstance(entry, dict) or not entry.get("local_path"):
                errors.append(f"{rel}: malformed provenance file entry")
                continue
            target = bundle / entry["local_path"]
            if not target.is_file():
                errors.append(f"{rel}: missing payload '{entry['local_path']}'")
                continue
            declared.add(target.resolve())
            role = entry.get("role")
            if role:
                role_paths[role] = target
            if sha256(target) != entry.get("sha256"):
                errors.append(f"{rel}: hash mismatch for '{entry['local_path']}'")
            if target.stat().st_size != entry.get("size"):
                errors.append(f"{rel}: size mismatch for '{entry['local_path']}'")
        actual = {path.resolve() for path in bundle.iterdir() if path.is_file() and path.name != "PROVENANCE.yaml"}
        if actual != declared:
            errors.append(f"{rel}: provenance payload list does not match bundle")
        before = role_paths.get("before-code")
        after = role_paths.get("after-code")
        diff = role_paths.get("unified-diff")
        if before and after and diff and before.is_file() and after.is_file() and diff.is_file():
            expected = "".join(
                difflib.unified_diff(
                    before.read_text(encoding="utf-8").splitlines(keepends=True),
                    after.read_text(encoding="utf-8").splitlines(keepends=True),
                    fromfile=before.name,
                    tofile=after.name,
                )
            )
            if diff.read_text(encoding="utf-8") != expected:
                errors.append(f"{rel}: changes.diff does not reproduce the local source pair")
    return errors


def ledger_errors(root: Path) -> list[str]:
    errors = []
    ledger = root / "data" / "trace-import-ledger.jsonl"
    if not ledger.is_file():
        return ["data/trace-import-ledger.jsonl: missing import ledger"]
    input_ids = set()
    for number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"data/trace-import-ledger.jsonl:{number}: invalid JSON")
            continue
        input_id = row.get("input_id")
        if input_id in input_ids:
            errors.append(f"data/trace-import-ledger.jsonl:{number}: duplicate input_id")
        input_ids.add(input_id)
        if row.get("disposition") == "accepted":
            expected = (
                root / "sources" / "experiments" / f"{row['task_family']}.md",
                root / "wiki" / "kernels" / f"{row['wiki_id']}.md",
                root / row["artifact_path"],
            )
            if not all(path.exists() for path in expected):
                errors.append(f"data/trace-import-ledger.jsonl:{number}: accepted outputs are incomplete")
        elif row.get("disposition") == "duplicate":
            if row.get("canonical_input_id") not in input_ids:
                # The deterministic ledger is sorted by input id, so validate
                # canonical membership after all rows have been parsed below.
                pass
        elif row.get("disposition") != "rejected":
            errors.append(f"data/trace-import-ledger.jsonl:{number}: invalid disposition")
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    for number, row in enumerate(rows, 1):
        if row.get("disposition") == "duplicate" and row.get("canonical_input_id") not in input_ids:
            errors.append(f"data/trace-import-ledger.jsonl:{number}: duplicate target is absent")
    return errors


def check_repository(root: Path, forbidden: list[str]) -> list[str]:
    root = root.resolve(strict=True)
    errors = []
    for path in iter_repo_paths(root):
        if not path.is_symlink():
            continue
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            errors.append(f"{path.relative_to(root)}: broken symlink")
            continue
        if root != resolved and root not in resolved.parents:
            errors.append(f"{path.relative_to(root)}: symlink resolves outside KernelWiki")

    for path, text in text_files(root):
        rel = path.relative_to(root)
        for token in forbidden:
            if token and token in text:
                errors.append(f"{rel}: contains forbidden source token")
        if rel.parts[:2] in {
            ("sources", "experiments"),
            ("artifacts", "experiments"),
        } or (rel.parts and rel.parts[0] == "wiki" and path.name.startswith("kernel-trace-")):
            for label, pattern in GENERATED_LOCATOR_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{rel}: contains {label}")
    errors.extend(local_link_errors(root))
    errors.extend(page_and_path_errors(root))
    errors.extend(experiment_errors(root))
    errors.extend(ledger_errors(root))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forbid-root", action="append", default=[], help="Absolute source root forbidden in repository text")
    parser.add_argument("--forbid-token", action="append", default=[], help="Additional source-repository token to forbid")
    args = parser.parse_args()
    forbidden = []
    for value in [*args.forbid_root, *args.forbid_token]:
        if value and value not in forbidden:
            forbidden.append(value)
    errors = check_repository(REPO_ROOT, forbidden)
    if errors:
        print(f"Self-containment check failed with {len(errors)} error(s):")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1
    print("Self-containment check passed.")
    print("All experiment artifacts, links, source IDs, hashes, diffs, and symlinks resolve inside KernelWiki.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
