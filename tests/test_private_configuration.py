"""Synthetic tests for loading private decisions without embedding study IDs."""
from pathlib import Path
import io
import tempfile
import unittest
from unittest.mock import patch

from gp_pipeline.aggregation.analysis_exclusions import AnalysisExclusions, resolve_analysis_exclusions
from gp_pipeline.aggregation.cohorts import build_parser


class PrivateConfigurationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "rules.local.toml"

    def test_explicit_choice_is_required_and_missing_file_does_not_disable_rules(self):
        with self.assertRaisesRegex(ValueError, "Supply exclusions_config"):
            resolve_analysis_exclusions(None)
        with self.assertRaisesRegex(ValueError, "not both"):
            resolve_analysis_exclusions(self.path, no_manual_exclusions=True)
        with self.assertRaises(FileNotFoundError):
            resolve_analysis_exclusions(self.path)
        self.assertEqual(resolve_analysis_exclusions(None, no_manual_exclusions=True), AnalysisExclusions())

    def test_file_normalizes_ids_and_keeps_cohort_policies_separate(self):
        self.path.write_text('[manual_exclusions]\nrmssd_sdnn_ids = ["0910001.0"]\nsleep_circadian_ids = ["920001"]\n')
        policy = resolve_analysis_exclusions(self.path)
        self.assertEqual(policy.rmssd_sdnn_ids, ("910001",))
        self.assertEqual(policy.sleep_circadian_ids, ("920001",))

    def test_rejects_misspelled_fields_wrong_types_invalid_and_duplicate_ids(self):
        cases = [
            '[manual_exclusion]\nrmssd_sdnn_ids = []\nsleep_circadian_ids = []',
            '[manual_exclusions]\nrmssd_sdnn_ids = []',
            '[manual_exclusions]\nrmssd_sdnn_ids = []\nsleep_circadian_ids = []\nextra = []',
            '[manual_exclusions]\nrmssd_sdnn_ids = [910001]\nsleep_circadian_ids = []',
            '[manual_exclusions]\nrmssd_sdnn_ids = ["invalid"]\nsleep_circadian_ids = []',
            '[manual_exclusions]\nrmssd_sdnn_ids = ["910001", "0910001.0"]\nsleep_circadian_ids = []',
        ]
        for case in cases:
            with self.subTest(case=case):
                self.path.write_text(case)
                with self.assertRaises(ValueError):
                    AnalysisExclusions.from_toml(self.path)

    def test_cli_requires_one_choice(self):
        parser = build_parser()
        for args in ([], ["--exclusions-config", "private.local.toml", "--no-manual-exclusions"]):
            with self.subTest(args=args), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(args)
        self.assertTrue(parser.parse_args(["--no-manual-exclusions"]).no_manual_exclusions)
        self.assertEqual(parser.parse_args(["--exclusions-config", "private.local.toml"]).exclusions_config,
                         "private.local.toml")
