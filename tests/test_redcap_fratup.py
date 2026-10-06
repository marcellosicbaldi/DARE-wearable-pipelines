"""Synthetic tests for external FRAT-up CSV linkage; no algorithm or real data."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import pandas as pd

from fallspredict_gp_pipeline.aggregation.redcap import build_parser, process_redcap, write_redcap_exports
from fallspredict_gp_pipeline.aggregation.overall import build_parser as build_overall_parser
from fallspredict_gp_pipeline.redcap.fratup import DEFAULT_FRATUP_DIR, attach_fratup
from fallspredict_gp_pipeline.redcap.schema import NON_SENSOR_COLUMNS
from test_redcap import baseline


class FratupTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.clinical = pd.DataFrame({"subject": ["0007", "0012"], "sex": ["F", "M"],
                                      "age": [75., 80.], "living_alone": [1., 0.],
                                      "fratup": [float("nan"), float("nan")]})
        self.identities = pd.DataFrame({"subject": ["0007", "0012"], "record_id": ["a", "b"]})
        self.results = pd.DataFrame({"record_id": ["b", "a"], "patient_id": ["12.0", "0007"],
                                     "fratup": [0.4, 0.2]})
        self.inputs = pd.DataFrame({"patient_id": ["12", "7"], "sex": [0, 1], "age": [80., 75.], "livingalone": [0, 1],
                                    "walkingaiduse": [1, 0]})

    def save(self, inputs=None, results=None, plural=False):
        (self.inputs if inputs is None else inputs).to_csv(self.root / "fratup_input_BO_T0.csv", index=False)
        (self.results if results is None else results).to_csv(self.root / ("results_BO_T0.csv" if plural else "result_BO_T0.csv"), index=False)

    def load(self):
        return attach_fratup(self.clinical, self.identities, self.root)

    def test_input_ids_link_values_and_are_preserved_without_becoming_risk_factors(self):
        self.save()
        with warnings.catch_warnings(record=True) as caught:
            imported = self.load()
        self.assertFalse(caught)
        self.assertEqual(imported.clinical.fratup.tolist(), [0.2, 0.4])
        self.assertEqual(imported.clinical.fratup_input_walkingaiduse.tolist(), [0., 1.])
        self.assertNotIn("fratup_input_age", imported.clinical)
        self.assertEqual(imported.inputs.subject.tolist(), ["0012", "0007"])
        self.assertEqual(imported.inputs.source_row.tolist(), [1, 2])
        self.assertEqual(imported.inputs.patient_id.tolist(), ["12", "7"])
        self.assertNotIn("fratup_input_patient_id", imported.clinical)
        self.assertEqual(imported.audit.action.eq("already_present").sum(), 3)

    def test_keyed_inputs_can_be_sorted_independently_and_use_plural_filename(self):
        inputs = self.inputs.assign(record_id=["b", "a"]).iloc[::-1]
        self.save(inputs=inputs, plural=True)
        # Files for other visits/cohorts never enter the BO T0 import.
        for name in ("result_BO_T1.csv", "result_RA_T0.csv", "fratup_input_RA_T0.csv"):
            (self.root / name).write_text("invalid unrelated file")
        imported = self.load()
        self.assertEqual(imported.clinical.fratup.tolist(), [0.2, 0.4])
        self.assertEqual(imported.clinical.fratup_input_walkingaiduse.tolist(), [0., 1.])
        self.assertIn("participant_ids", imported.audit.action.tolist())

    def test_keyed_conflicting_input_is_preserved_without_overwriting_clinical(self):
        inputs = self.inputs.assign(subject=["0012", "0007"])
        inputs.loc[1, "age"] = 76
        self.save(inputs=inputs)
        imported = self.load()
        self.assertEqual(imported.clinical.age.tolist(), [75., 80.])
        self.assertEqual(imported.clinical.fratup_input_age.tolist(), [76., 80.])
        audit = imported.audit.set_index("source_field")
        self.assertEqual(audit.loc["age", "different_rows"], 1)
        self.assertEqual(imported.inputs.source_subject.tolist(), ["0012", "0007"])

    def test_idless_inputs_are_rejected_even_when_rows_and_demographics_match(self):
        self.save(inputs=self.inputs.drop(columns="patient_id"))
        with self.assertRaisesRegex(ValueError, "participant ID column required"):
            self.load()

    def test_input_and_result_participant_sets_must_agree(self):
        self.save(inputs=self.inputs.iloc[:1])
        with self.assertRaisesRegex(ValueError, "participant sets disagree"):
            self.load()

    def test_ids_link_inputs_without_demographic_columns(self):
        self.save(inputs=self.inputs[["patient_id", "walkingaiduse"]].iloc[::-1])
        self.assertEqual(self.load().clinical.fratup_input_walkingaiduse.tolist(), [0., 1.])

    def test_input_identity_errors_are_rejected_without_positional_fallback(self):
        variants = [self.inputs.assign(patient_id=[None, None]),
                    self.inputs.assign(patient_id=["9999", "7"]),
                    self.inputs.assign(record_id=["a", "b"])]
        for inputs in variants:
            with self.subTest(inputs_shape=inputs.shape):
                self.save(inputs=inputs)
                with self.assertRaises(ValueError):
                    self.load()

    def test_duplicate_unknown_conflicting_or_missing_ids_are_rejected(self):
        variants = [self.results.iloc[[0, 0]], self.results.assign(patient_id=["9999", "7"]),
                    self.results.assign(patient_id=["7", "12"]),
                    self.results.assign(record_id=["unknown", "a"]),
                    self.results.assign(record_id=[None, None], patient_id=[None, None]),
                    self.results.assign(patient_id=["12.5", "7"])]
        for results in variants:
            with self.subTest(results_shape=results.shape):
                self.save(results=results)
                with self.assertRaises(ValueError):
                    self.load()
        self.save(inputs=self.inputs.assign(patient_id=["7", "7"]))
        with self.assertRaises(ValueError):
            self.load()

    def test_id_alias_means_record_id_and_blank_patient_id_can_resolve_from_record(self):
        self.save(inputs=self.inputs.assign(id=["b", "a"]),
                  results=self.results.assign(patient_id=[None, None]))
        self.assertEqual(self.load().clinical.fratup.tolist(), [0.2, 0.4])
        self.save(results=pd.DataFrame({"id": ["12", "7"], "fratup": [0.4, 0.2]}))
        with self.assertRaisesRegex(ValueError, "outside"):
            self.load()

    def test_missing_scores_remain_missing_and_invalid_probabilities_fail(self):
        self.save(inputs=self.inputs.assign(patient_id=["12", "7"]).iloc[:1], results=self.results.iloc[:1])
        with self.assertWarnsRegex(UserWarning, "no imported score"):
            imported = self.load()
        self.assertTrue(pd.isna(imported.clinical.fratup.iloc[0]))
        for value in ("oops", -0.1, 20, "inf"):
            self.save(results=self.results.assign(fratup=[value, 0.2]))
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load()
        self.save(inputs=self.inputs.assign(patient_id=["12", "7"]),
                  results=self.results.assign(fratup=[None, 0.2]))
        with self.assertWarns(UserWarning):
            imported = self.load()
        self.assertTrue(pd.isna(imported.clinical.fratup.iloc[1]))

    def test_ambiguous_results_missing_inputs_and_visit_mismatch_fail(self):
        self.save()
        self.save(plural=True)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.load()
        (self.root / "results_BO_T0.csv").unlink()
        self.save(results=self.results.assign(visit="T1"))
        with self.assertRaisesRegex(ValueError, "must be T0"):
            self.load()
        (self.root / "fratup_input_BO_T0.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_explicit_disable_and_cli_defaults(self):
        imported = attach_fratup(self.clinical, self.identities, None)
        pd.testing.assert_frame_equal(imported.clinical, self.clinical)
        for parser in (build_parser(), build_overall_parser()):
            args = ["--output-dir", str(self.root)]
            self.assertEqual(parser.parse_args(args).fratup_dir, str(DEFAULT_FRATUP_DIR))
            self.assertIsNone(parser.parse_args([*args, "--no-fratup"]).fratup_dir)

    def test_pipeline_exports_and_overall_merge_include_scores_and_extra_inputs(self):
        from fallspredict_gp_pipeline.aggregation.overall import aggregate_all, write_all_exports
        self.save(inputs=self.inputs.assign(patient_id=["12", "7"]))
        second = baseline("b", "12")
        second.update(patient_gender="1", patient_age="80", living_alonef_b="0")
        source = self.root / "redcap.csv"
        pd.DataFrame([baseline(), second]).to_csv(source, index=False)
        processed = process_redcap(source, fratup_dir=self.root)
        self.assertEqual(tuple(processed.clinical.columns[:len(NON_SENSOR_COLUMNS)]), NON_SENSOR_COLUMNS)
        self.assertEqual(processed.clinical.fratup.tolist(), [0.2, 0.4])
        self.assertEqual(processed.mapping.set_index("variable").loc["fratup", "status"], "external_csv")
        sensors = pd.DataFrame({"subject": ["7"], "visit": ["T0"], "sleep_value": [10]})
        sensor_path = self.root / "sensors.csv"
        sensors.to_csv(sensor_path, index=False)
        paths = write_redcap_exports(source, output_dir=self.root / "out", fratup_dir=self.root, sensor_csv=sensor_path)
        self.assertTrue({"fratup_inputs", "fratup_import", "clinical_sensors_T0"}.issubset(paths))
        combined = pd.read_csv(paths["clinical_sensors_T0"])
        self.assertEqual(combined.fratup.tolist(), [0.2, 0.4])
        self.assertEqual(combined.fratup_input_walkingaiduse.tolist(), [0., 1.])
        with patch("fallspredict_gp_pipeline.aggregation.overall.aggregate_sleep", return_value=sensors), \
             patch("fallspredict_gp_pipeline.aggregation.overall._build_non_sleep_domain_frames", return_value=[]), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore")
            merged = aggregate_all(redcap_csv=source, fratup_dir=self.root)
            self.assertEqual(merged.fratup.tolist(), [0.2, 0.4])
            outputs = write_all_exports(output_dir=self.root / "overall", redcap_csv=source, fratup_dir=self.root)
            for _, frame in outputs.values():
                self.assertEqual(frame.fratup.tolist(), [0.2, 0.4])


if __name__ == "__main__":
    unittest.main()
