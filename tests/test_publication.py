"""Publication guard tests use invented strings and temporary repositories."""
from pathlib import Path
import importlib.util
import json
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_publication.py"
spec = importlib.util.spec_from_file_location("publication_check", SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class PublicationTests(unittest.TestCase):
    def test_rejects_data_and_private_configs_even_with_harmless_contents(self):
        for name in ("measurements.csv", "source.OMX", "src/gp_pipeline/outputs/report.md",
                     "configs/settings.local.toml", "configs/settings.toml", "reference.pdf"):
            with self.subTest(name=name):
                self.assertTrue(guard.check_content(name, b"example"))

    def test_checks_notebook_execution_state_and_escaped_source_paths(self):
        notebook = {"metadata": {"language_info": {"name": "python"}, "kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [{"cell_type": "code", "metadata": {}, "source": ["print(1)"],
                       "execution_count": None, "outputs": []}]}
        self.assertFalse(guard.check_content("notebooks/example.ipynb", json.dumps(notebook).encode()))
        notebook["cells"][0]["outputs"] = [{"output_type": "stream", "text": "synthetic result"}]
        self.assertTrue(guard.check_content("notebooks/example.ipynb", json.dumps(notebook).encode()))
        notebook["cells"][0]["outputs"] = []
        notebook["cells"][0]["source"] = ["/" + "Users/fictional-person/private"]
        self.assertTrue(guard.check_content("notebooks/example.ipynb", json.dumps(notebook).encode()))

    def test_example_rules_cannot_contain_ids(self):
        template = '[manual_exclusions]\nrmssd_sdnn_ids = []\nsleep_circadian_ids = []'
        self.assertFalse(guard.check_content("configs/analysis_exclusions.example.toml", template.encode()))
        self.assertTrue(guard.check_content("configs/analysis_exclusions.example.toml",
                                          template.replace('ids = []', 'ids = ["910001"]', 1).encode()))

    def test_scans_the_index_even_when_working_file_is_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            readme = root / "README.md"
            readme.write_text("/" + "Users/fictional-person/private")
            subprocess.run(["git", "-C", str(root), "add", "README.md"], check=True)
            readme.write_text("Public example")
            _, issues = guard.audit(root)
            self.assertTrue(any(item.startswith("index:README.md:") for item in issues))
            self.assertFalse(any(item.startswith("working:README.md:") for item in issues))
            self.assertFalse(any("fictional-person" in item for item in issues))

    def test_force_added_ignored_files_and_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text("*.csv\n")
            (root / "data.csv").write_text("synthetic\n")
            (root / "link.md").symlink_to(root / "data.csv")
            subprocess.run(["git", "-C", str(root), "add", "-f", "data.csv", "link.md"], check=True)
            _, issues = guard.audit(root)
            self.assertTrue(any("index:data.csv:" in item for item in issues))
            self.assertTrue(any("index:link.md:" in item for item in issues))
            self.assertTrue(any("working:link.md:" in item for item in issues))
