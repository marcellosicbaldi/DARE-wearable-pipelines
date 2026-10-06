#!/usr/bin/env python3
"""Check working/index contents without printing potential sensitive values."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


MAX_BYTES = 1_000_000
ALLOWED_SUFFIXES = {".py", ".md", ".toml", ".txt", ".ipynb", ".yml", ".yaml"}
ALLOWED_NAMES = {".gitignore", ".gitattributes", "MANIFEST.in", "LICENSE", "NOTICE", ".python-version", "uv.lock"}
PRIVATE_PARTS = {"data", "bronze", "silver", "gold", "outputs", "private", "logs",
                 "legacy", "playground", "debug", "other_pipelines", ".vscode", ".idea",
                 "__pycache__", ".ipynb_checkpoints"}
PATTERNS = {
    "workstation path": re.compile(r"/(?:Users|Volumes|home)/[^\s\"'<>]+"),
    "Windows user path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+", re.I),
    "network share": re.compile(r"(?<!:)//[\w.-]+\.[a-z]{2,}/|\\\\[\w.-]+\\", re.I),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{30,})\b"),
    "credential assignment": re.compile(r"(?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*[\"'][^\"'\n]{12,}[\"']", re.I),
}


def check_content(name: str, payload: bytes) -> list[str]:
    path = PurePosixPath(name)
    issues = []
    if any(part.lower() in PRIVATE_PARTS for part in path.parts):
        issues.append("private/generated directory")
    if path.name not in ALLOWED_NAMES and path.suffix.lower() not in ALLOWED_SUFFIXES:
        issues.append("file type is not in the publication allowlist")
    if ".local." in path.name or ".private." in path.name or path.name.startswith(".env"):
        issues.append("private configuration")
    if "configs" in path.parts and not (path.name.endswith(".example.toml") or path.suffix == ".md"):
        issues.append("only example configurations may be published")
    if len(payload) > MAX_BYTES:
        return issues + ["file exceeds publication size limit"]
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return issues + ["binary content"]
    # Decode notebook strings before scanning so JSON escaping cannot hide paths.
    scan_text = text
    if path.suffix == ".ipynb":
        try:
            notebook = json.loads(text)
            scan_text = "\n".join("".join(c.get("source", [])) for c in notebook["cells"])
            if set(notebook.get("metadata", {})) - {"kernelspec", "language_info"}:
                issues.append("notebook execution/custom metadata")
            if notebook.get("metadata", {}).get("language_info", {}) != {"name": "python"}:
                issues.append("notebook environment metadata must be reduced to language name")
            expected_kernel = {"display_name": "Python 3", "language": "python", "name": "python3"}
            if notebook.get("metadata", {}).get("kernelspec", {}) != expected_kernel:
                issues.append("notebook kernel metadata must be generic")
            for index, cell in enumerate(notebook["cells"]):
                if cell.get("outputs") or cell.get("execution_count") is not None or cell.get("attachments") or cell.get("metadata"):
                    issues.append(f"notebook cell {index} has output, execution state, attachment or metadata")
        except (ValueError, KeyError, TypeError, AttributeError):
            issues.append("invalid notebook structure")
    for label, pattern in PATTERNS.items():
        match = pattern.search(scan_text)
        if match:
            issues.append(f"{label} at text line {scan_text[:match.start()].count(chr(10)) + 1}")
    if path.name.endswith(".example.toml"):
        try:
            config = tomllib.loads(text)
            if "manual_exclusions" in config and any(config["manual_exclusions"].values()):
                issues.append("example exclusions contain identifiers")
            for section in config.values():
                if isinstance(section, dict):
                    if section.get("participant") not in (None, "900001") or section.get("participant_ids"):
                        issues.append("example participant selection is not synthetic/empty")
        except (ValueError, TypeError, AttributeError):
            issues.append("invalid example configuration")
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.endswith("_OUTLIER_IDS") for t in node.targets):
                    issues.append("embedded study exclusion list; use private configuration")
        except SyntaxError:
            issues.append("invalid Python syntax")
    return issues


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def audit(root: Path) -> tuple[int, list[str]]:
    """Inspect all Git-visible working files and every staged blob."""
    findings = []
    if (root / ".git").exists():
        names = set(git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").decode().split("\0")) - {""}
        staged = git(root, "ls-files", "--stage", "-z").decode().split("\0")
        for entry in filter(None, staged):
            meta, name = entry.split("\t", 1)
            mode, blob, stage = meta.split()
            if mode not in {"100644", "100755"} or stage != "0":
                findings.append(f"index:{name}: unsupported mode or unresolved merge")
                continue
            if int(git(root, "cat-file", "-s", blob)) > MAX_BYTES:
                findings.append(f"index:{name}: file exceeds publication size limit")
                continue
            findings.extend(f"index:{name}: {reason}" for reason in check_content(name, git(root, "cat-file", "blob", blob)))
    else:
        # Before git init, inspect the entire tree. Private additions must not be
        # copied into this publication candidate in the first place.
        names = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    for name in sorted(names):
        file = root / name
        if file.is_symlink():
            findings.append(f"working:{name}: symlink is not allowed")
        elif file.is_file():
            if file.stat().st_size > MAX_BYTES:
                findings.append(f"working:{name}: file exceeds publication size limit")
            else:
                findings.extend(f"working:{name}: {reason}" for reason in check_content(name, file.read_bytes()))
    return len(names), findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    count, findings = audit(args.root.resolve())
    if findings:
        print("Publication check failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(f"Publication check passed: {count} candidate files; working tree and available Git index checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
