"""Synthetic cohort-union and local export tests; never use participant files."""

from pathlib import Path
import io
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from fallspredict_gp_pipeline.aggregation.cohorts import (
    build_parser as build_cohorts_parser,
    build_combined_dataset, combine_cohort_frames,
    load_sensor_exports, read_sensor_export, write_combined_exports,
)
from fallspredict_gp_pipeline.aggregation.redcap import build_parser, merge_redcap_with_sensors, process_redcap
from fallspredict_gp_pipeline.aggregation.analysis_exclusions import EXCLUDED_ANALYSIS_COLUMNS
from fallspredict_gp_pipeline.redcap.cohorts import default_redcap_csv
from test_redcap import baseline


class CombinedCohortsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.bo = self.root / "BO"
        self.ra = self.root / "RA"
        self.fratup = self.root / "fratup"
        for folder in (self.bo, self.ra, self.fratup):
            folder.mkdir()

    def csv(self, path, rows):
        pd.DataFrame(rows).fillna("").to_csv(path, index=False)
        return path

    def fixture(self):
        # Both cohorts deliberately contain patient 7 and record 'a'.
        bo = [dict(baseline("a", "7"), bologna_extra="BO only"), baseline("b", "8")]
        ra = [dict(baseline("a", "7"), patient_age="81", ravenna_extra="RA baseline"),
              {"record_id": "a", "patient_id": "", "redcap_event_name": "mese_1_arm_1", "cadute_mese": "1",
               "ravenna_extra": "first", "date_fall_c5": "2025-05-12", "domande_quinta_caduta_complete": "2"},
              {"record_id": "a", "patient_id": "", "redcap_event_name": "mese_2_arm_1", "cadute_mese": "0",
               "ravenna_extra": "second", "date_fall_c5": "", "domande_quinta_caduta_complete": "0"}]
        self.csv(self.bo / "redcap.csv", bo)
        self.csv(self.ra / "redcap.csv", ra)
        self.csv(self.bo / "overall_T0_sleep_mean.csv", [{"subject": "7", "visit": "T0", "sleep_value": 400, "hrv_only": 50,
                                                        "hrv_n_valid_nights": 3}])
        self.csv(self.ra / "sleep_T0_mean.csv", [{"subject": "7", "visit": "T0", "sleep_value": 420, "ra_sensor_only": 3}])
        self.csv(self.ra / "sleep_T0_median.csv", [{"subject": "7", "visit": "T0", "sleep_value": 410, "ra_sensor_only": 4}])
        self.csv(self.ra / "gait_T0_mean.csv", [{"subject": "7", "visit": "T0", "gait_value": 10, "gait_n_valid_days": 3},
                                               {"subject": "9", "visit": "T0", "gait_value": 20, "gait_n_valid_days": 3}])
        self.csv(self.ra / "activity_intensity_T0.csv", [{"subject": "7", "visit": "T0", "activity_value": 5,
                                                       "activity_n_valid_days": 3}])
        for cohort, ids, records, scores in (("BO", ["7", "8"], ["a", "b"], [0.1, 0.2]), ("RA", ["7"], ["a"], [0.6])):
            self.csv(self.fratup / f"result_{cohort}_T0.csv", {"record_id": records, "patient_id": ids, "fratup": scores})
            self.csv(self.fratup / f"fratup_input_{cohort}_T0.csv", {"record_id": records, "patient_id": ids, "walkingaiduse": [1] * len(ids)})
        return {"bologna_sensor_dir": self.bo, "ravenna_sensor_dir": self.ra,
                "bologna_redcap_csv": self.bo / "redcap.csv", "ravenna_redcap_csv": self.ra / "redcap.csv",
                "fratup_dir": self.fratup, "no_manual_exclusions": True}

    def test_end_to_end_union_preserves_cohort_ids_without_raw_event_columns(self):
        result = build_combined_dataset(**self.fixture())
        data = result.data.set_index(["group", "subject", "visit"])
        self.assertEqual(len(data), 4)
        self.assertEqual(data.loc[("BO", "0007", "T0"), "age"], 75)
        self.assertEqual(data.loc[("RA", "0007", "T0"), "age"], 81)
        self.assertEqual(data.loc[("BO", "0007", "T0"), "fratup"], 0.1)
        self.assertEqual(data.loc[("RA", "0007", "T0"), "fratup"], 0.6)
        self.assertTrue(data.loc["RA", "hrv_only"].isna().all())
        self.assertTrue(data.loc["BO", "ra_sensor_only"].isna().all())
        self.assertTrue(pd.isna(data.loc[("BO", "0008", "T0"), "sleep_value"]))
        self.assertEqual(data.loc[("RA", "0009", "T0"), "has_redcap"], 0)
        self.assertTrue(pd.isna(data.loc[("RA", "0009", "T0"), "fratup"]))
        self.assertFalse(any(c.startswith("redcap__") for c in data))
        self.assertNotIn("bologna_extra", data)
        self.assertNotIn("ravenna_extra", data)
        self.assertIn("fall_occurred_m01", data)
        self.assertEqual(result.redcap["RA"].events.ravenna_extra.tolist(), ["RA baseline", "first", "second"])
        self.assertEqual(result.cohorts["BO"].columns.tolist(), result.cohorts["RA"].columns.tolist())
        self.assertEqual(result.cohorts["RA"].columns.tolist(), result.data.columns.tolist())
        fifth = result.redcap["RA"].falls.query("fall_slot == 5")
        self.assertEqual(fifth.verified.tolist(), [1])
        coverage = result.column_coverage.set_index("variable")
        self.assertFalse(coverage.loc["ra_sensor_only", "BO_column_present"])
        self.assertTrue(coverage.loc["ra_sensor_only", "RA_column_present"])
        summary = result.summary.set_index("group")
        self.assertEqual(summary.loc["BO", "clinical_only"], 1)
        self.assertEqual(summary.loc["RA", "sensor_only"], 1)

    def test_writer_exports_identical_headers_nan_values_and_cohort_provenance(self):
        paths = write_combined_exports(output_dir=self.root / "output", **self.fixture())
        for name, path in paths.items():
            self.assertEqual(path.name, f"fallspredict_{name}.csv")
        frames = [pd.read_csv(paths[name], dtype={"subject": "string"}) for name in (
            "combined_T0", "bologna_T0", "ravenna_T0")]
        self.assertEqual(frames[0].columns.tolist(), frames[1].columns.tolist())
        self.assertEqual(frames[1].columns.tolist(), frames[2].columns.tolist())
        self.assertTrue(frames[2].hrv_only.isna().all())
        self.assertTrue(all(not any(c.startswith("redcap__") for c in frame) for frame in frames))
        self.assertNotIn("extra_redcap_fields", paths)
        events = pd.read_csv(paths["redcap_events"])
        self.assertEqual(events.loc[events.group.eq("RA"), "ravenna_extra"].tolist(), ["RA baseline", "first", "second"])
        self.assertIn("NaN", paths["combined_T0"].read_text())
        for name in ("redcap_events", "redcap_falls", "redcap_mapping", "redcap_fratup_inputs", "redcap_fratup_import"):
            self.assertIn("group", pd.read_csv(paths[name]).columns)
        source = pd.read_csv(paths["source_files_T0"])
        self.assertEqual(source.loc[source.role.eq("sensor")].group.value_counts().to_dict(), {"RA": 3, "BO": 1})

    def test_save_subject_id_argument_defaults_true_and_accepts_explicit_booleans(self):
        parser = build_cohorts_parser()
        self.assertIs(parser.parse_args(["--no-manual-exclusions"]).save_subject_id, True)
        for value, expected in (("TRUE", True), ("true", True), ("FALSE", False), ("False", False)):
            self.assertIs(parser.parse_args(["--no-manual-exclusions", "--save-subject-id", value]).save_subject_id, expected)
        for args in (["--save-subject-id", "invalid"], ["--save-subject-id"]):
            with self.subTest(args=args), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["--no-manual-exclusions", *args])
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            write_combined_exports(save_subject_id="FALSE")

    def test_omitting_ids_preserves_values_and_removes_aliases_from_all_exports(self):
        kwargs = self.fixture()
        for path in self.fratup.glob("fratup_input_*.csv"):
            inputs = pd.read_csv(path, dtype="string")
            inputs["subject"] = inputs["patient_id"]
            inputs["id"] = inputs["record_id"]
            inputs.to_csv(path, index=False)
        (self.bo / "overall_T0_sleep_median.csv").write_bytes((self.bo / "overall_T0_sleep_mean.csv").read_bytes())
        originals = {p: p.read_bytes() for folder in (self.bo, self.ra, self.fratup) for p in folder.glob("*.csv")}
        identifiers = {"subject", "subject_id", "patient_id", "record_id", "id", "source_subject"}
        for method in ("mean", "median"):
            with self.subTest(method=method):
                suffix = "" if method == "mean" else "_sleep_median"
                paths = write_combined_exports(output_dir=self.root / "output", sleep_method=method, **kwargs)
                with_ids = {name: pd.read_csv(path, dtype="string", keep_default_na=False) for name, path in paths.items()}
                self.assertTrue({"subject", "source_subject", "id", "record_id", "patient_id"}.issubset(with_ids["redcap_fratup_inputs"]))
                without_ids = write_combined_exports(
                    output_dir=self.root / "output", sleep_method=method, save_subject_id=False, **kwargs,
                )
                self.assertEqual(paths, without_ids)
                for name, path in without_ids.items():
                    saved = pd.read_csv(path, dtype="string", keep_default_na=False)
                    self.assertFalse(identifiers.intersection(saved.columns))
                    expected = with_ids[name].drop(columns=list(identifiers), errors="ignore")
                    if name == f"column_coverage_T0{suffix}":
                        expected = expected.loc[~expected.variable.isin(identifiers)].reset_index(drop=True)
                    pd.testing.assert_frame_equal(saved, expected)
                self.assertIn("subject", with_ids[f"combined_T0{suffix}"])
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)

    def test_sensor_mean_and_median_are_alternatives_and_overall_takes_precedence(self):
        self.fixture()
        mean, _ = load_sensor_exports(self.ra, cohort="RA", sleep_method="mean")
        median, _ = load_sensor_exports(self.ra, cohort="RA", sleep_method="median")
        self.assertEqual(mean.set_index("subject").loc["0007", "sleep_value"], 420)
        self.assertEqual(median.set_index("subject").loc["0007", "sleep_value"], 410)
        overall = self.csv(self.ra / "overall_T0_sleep_mean.csv", [{"subject": "7", "visit": "T0", "sleep_value": 999}])
        frame, paths = load_sensor_exports(self.ra, cohort="RA")
        self.assertEqual(paths, [overall])
        self.assertEqual(frame.sleep_value.tolist(), [999])
        self.assertNotIn("gait_value", frame)

    def test_sensor_exports_reject_duplicates_wrong_visit_group_and_overlapping_fields(self):
        self.fixture()
        path = self.ra / "gait_T0_mean.csv"
        for rows in ([{"subject": "7", "visit": "T1"}], [{"subject": "7", "visit": "T0", "group": "BO"}],
                     [{"subject": "7", "visit": "T0"}, {"subject": "0007", "visit": "T0"}],
                     [{"subject": None, "visit": "T0"}]):
            self.csv(path, rows)
            with self.assertRaises(ValueError):
                read_sensor_export(path, cohort="RA")
        self.csv(path, [{"subject": "7", "visit": "T0", "sleep_value": 15}])
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            load_sensor_exports(self.ra, cohort="RA")

    def test_export_discovery_rejects_missing_and_ambiguous_files(self):
        self.fixture()
        (self.ra / "activity_intensity_T0.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            load_sensor_exports(self.ra, cohort="RA")
        (self.bo / "overall_T0_sleep_mean.xlsx").touch()
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            load_sensor_exports(self.bo, cohort="BO")

    def test_single_sheet_xlsx_reader_preserves_ids(self):
        # Mock the Excel engine; the real input workbook shapes/headers were
        # inspected separately without rerunning the participant aggregation.
        frame = pd.DataFrame({"subject": ["12.0"], "visit": ["T0"], "sensor": [3.]})
        with patch("fallspredict_gp_pipeline.aggregation.cohorts.pd.ExcelFile") as excel, \
             patch("fallspredict_gp_pipeline.aggregation.cohorts.pd.read_excel", return_value=frame):
            excel.return_value.__enter__.return_value.sheet_names = ["export.csv"]
            result = read_sensor_export(self.ra / "sleep.xlsx", cohort="RA")
            self.assertEqual(result.subject.tolist(), ["0012"])
            excel.return_value.__enter__.return_value.sheet_names = ["one", "two"]
            with self.assertRaisesRegex(ValueError, "exactly one sheet"):
                read_sensor_export(self.ra / "sleep.xlsx", cohort="RA")

    def test_multi_cohort_merge_never_matches_same_subject_across_groups(self):
        clinical = pd.DataFrame({"group": ["BO", "RA"], "subject": ["7", "7"], "age": [75, 81]})
        sensor = pd.DataFrame({"group": ["RA", "BO"], "subject": ["7", "7"], "sensor": [20, 10]})
        result = merge_redcap_with_sensors(clinical, sensor).set_index("group")
        self.assertEqual(result.loc["BO", "sensor"], 10)
        self.assertEqual(result.loc["RA", "sensor"], 20)
        with self.assertRaisesRegex(ValueError, "Both tables need group"):
            merge_redcap_with_sensors(clinical, sensor.drop(columns="group"))

    def test_removing_raw_expansion_preserves_participants_with_followup_only(self):
        kwargs = self.fixture()
        path = self.ra / "redcap.csv"
        events = pd.read_csv(path, dtype="string", keep_default_na=False)
        followup = pd.DataFrame([{"record_id": "c", "patient_id": "10", "redcap_event_name": "mese_1_arm_1",
                                  "cadute_mese": "0", "ravenna_extra": "retained in events"}])
        pd.concat([events, followup], ignore_index=True).fillna("").to_csv(path, index=False)
        data = build_combined_dataset(**kwargs).data.set_index(["group", "subject"])
        self.assertEqual(data.loc[("RA", "0010"), "has_redcap"], 1)
        self.assertEqual(data.loc[("RA", "0010"), "has_sensors"], 0)
        self.assertTrue(pd.isna(data.loc[("RA", "0010"), "age"]))
        self.assertFalse(any(c.startswith("redcap__") for c in data))

    def test_cohort_validation_and_ra_default_redcap_path(self):
        # Renaming display labels must not redirect existing clinical inputs.
        self.assertEqual(default_redcap_csv("BO").parent.name, "Bologna")
        self.assertIn("Ravenna", str(default_redcap_csv("RA")))
        self.assertEqual(build_parser().parse_args(["--cohort", "RA", "--output-dir", "unused"]).cohort, "RA")
        with self.assertRaises(ValueError):
            process_redcap(cohort="invalid", fratup_dir=None)
        self.fixture()
        with patch("fallspredict_gp_pipeline.aggregation.redcap.default_redcap_csv", return_value=self.ra / "redcap.csv") as default:
            result = process_redcap(cohort="RA", fratup_dir=self.fratup)
        default.assert_called_once_with("RA")
        self.assertEqual(result.clinical.group.tolist(), ["RA"])
        self.assertEqual(result.clinical.fratup.tolist(), [0.6])

    def test_missing_cohort_frame_and_duplicate_keys_are_rejected(self):
        frame = pd.DataFrame({"group": ["BO"], "subject": ["0007"], "visit": ["T0"]})
        with self.assertRaises(ValueError):
            combine_cohort_frames({"BO": frame})
        with self.assertRaises(ValueError):
            combine_cohort_frames({"BO": pd.concat([frame, frame]), "RA": frame.assign(group="RA")})

    def test_final_exclusions_apply_to_mean_median_and_both_cohort_exports(self):
        kwargs = self.fixture()
        private_config = self.root / "analysis_exclusions.local.toml"
        private_config.write_text('[manual_exclusions]\nrmssd_sdnn_ids = ["910001"]\nsleep_circadian_ids = ["920001"]\n')
        kwargs.pop("no_manual_exclusions")
        kwargs["exclusions_config"] = private_config
        # Only synthetic files are rewritten here. Reuse the fixture with one
        # listed participant per cohort and explicitly populated target metrics.
        for cohort, folder, subject in (("BO", self.bo, "910001"), ("RA", self.ra, "0920001")):
            for path in [*folder.glob("*.csv"), *self.fratup.glob(f"*_{cohort}_T0.csv")]:
                frame = pd.read_csv(path, dtype={"subject": "string", "patient_id": "string"})
                for key in ("subject", "patient_id"):
                    if key in frame:
                        frame[key] = frame[key].replace("7", subject)
                if "subject" in frame:
                    if cohort == "BO" or path.name.startswith("gait_"):
                        frame["gait_n_valid_days"] = frame["subject"].ne(subject).astype(int) * 3
                        frame["gait_n_days"] = 5
                        frame["gait_mean_valid_hours"] = 12.
                        frame["gait_wb_30__walking_speed_mps__avg"] = 0.8
                    if cohort == "BO":
                        frame["hrv_rmssd"] = 900.
                        frame["hrv_sdnn"] = 950.
                        frame["hrv_mean_hr"] = 65.
                        for metric in ("rmssd", "sdnn", "mean_hr"):
                            frame[f"hrv_{metric}_n_valid_nights"] = 3
                        for field in EXCLUDED_ANALYSIS_COLUMNS:
                            frame[field] = 1.
                    elif path.name.startswith("sleep_"):
                        frame["sleep_diary_tst_min"] = 420.
                        frame["hdcza_tst_min"] = 400.
                        frame["lower_back_tst_min"] = 410.
                        frame["circadian_IS"] = 0.8
                    elif path.name.startswith("activity_"):
                        frame["activity_day_mvpa_min_mean"] = 30.
                frame.to_csv(path, index=False)
        (self.bo / "overall_T0_sleep_median.csv").write_bytes((self.bo / "overall_T0_sleep_mean.csv").read_bytes())
        original_inputs = {p: p.read_bytes() for folder in (self.bo, self.ra, self.fratup) for p in folder.glob("*.csv")}
        for method in ("mean", "median"):
            with self.subTest(method=method):
                suffix = "" if method == "mean" else "_sleep_median"
                paths = write_combined_exports(output_dir=self.root / "output", sleep_method=method, **kwargs)
                data = pd.read_csv(paths[f"combined_T0{suffix}"], dtype={"subject": "string"})
                rows = data.set_index(["group", "subject"])
                bo = rows.loc[("BO", "910001")]
                ra = rows.loc[("RA", "920001")]
                self.assertTrue(bo[["hrv_rmssd", "hrv_sdnn"]].isna().all())
                self.assertEqual(bo.hrv_mean_hr, 65.)
                self.assertEqual(bo.rmssd_sdnn_exclusion, 1)
                self.assertEqual(bo.rmssd_sdnn_exclusion_method, "visual inspection")
                self.assertTrue(ra[["sleep_diary_tst_min", "hdcza_tst_min", "lower_back_tst_min", "circadian_IS", "activity_day_mvpa_min_mean"]].isna().all())
                self.assertEqual(ra.sleep_circadian_exclusion, 1)
                self.assertEqual(ra.sleep_circadian_exclusion_method, "visual inspection")
                self.assertTrue(pd.isna(ra.gait_value))
                for row in (bo, ra):
                    self.assertTrue(pd.isna(row.gait_wb_30__walking_speed_mps__avg))
                    self.assertEqual(row.gait_n_valid_days, 0)
                    self.assertEqual(row.gait_n_days, 5)
                    self.assertEqual(row.gait_mean_valid_hours, 12.)
                self.assertEqual(rows.loc[("RA", "0009"), "gait_value"], 20)
                self.assertEqual(rows.loc[("RA", "0009"), "gait_wb_30__walking_speed_mps__avg"], 0.8)
                self.assertEqual(ra.has_sensors, 1)
                self.assertTrue(pd.notna(bo.gait_speed))
                self.assertTrue(pd.notna(ra.gait_speed))
                for name in ("combined", "bologna", "ravenna"):
                    exported = pd.read_csv(paths[f"{name}_T0{suffix}"])
                    self.assertEqual(exported.columns.tolist(), data.columns.tolist())
                    self.assertFalse(set(EXCLUDED_ANALYSIS_COLUMNS).intersection(exported.columns))
                    zero_gait = exported.gait_n_valid_days.eq(0)
                    self.assertTrue(exported.loc[zero_gait, "gait_wb_30__walking_speed_mps__avg"].isna().all())
                    for flag in ("rmssd_sdnn_exclusion", "sleep_circadian_exclusion"):
                        self.assertTrue(exported.loc[exported[flag].eq(0), f"{flag}_method"].isna().all())
                coverage = pd.read_csv(paths[f"column_coverage_T0{suffix}"]).set_index("variable")
                self.assertEqual(coverage.loc["hrv_rmssd", "BO_nonmissing_rows"], 0)
                self.assertEqual(coverage.loc["sleep_diary_tst_min", "RA_nonmissing_rows"], 0)
                self.assertEqual(coverage.loc["gait_wb_30__walking_speed_mps__avg", "BO_nonmissing_rows"], 0)
                self.assertEqual(coverage.loc["gait_wb_30__walking_speed_mps__avg", "RA_nonmissing_rows"], 1)
                self.assertFalse(set(EXCLUDED_ANALYSIS_COLUMNS).intersection(coverage.index))
                summary = pd.read_csv(paths[f"cohort_summary_T0{suffix}"]).set_index("group")
                self.assertEqual(summary.loc["BO", "rmssd_sdnn_exclusions"], 1)
                self.assertEqual(summary.loc["RA", "sleep_circadian_exclusions"], 1)
        for path, original in original_inputs.items():
            self.assertEqual(path.read_bytes(), original)

    def test_three_observation_minimum_applies_to_both_cohorts_and_export_variants(self):
        kwargs = self.fixture()
        measurements = {"gait_step_count": 2000., "sleep_diary_tst_min": 420.,
                        "hdcza_tst_min": 400., "lower_back_tst_min": 410.,
                        "hrv_rmssd": 30., "median_hr_day": 70., "median_hr_night": 60.,
                        "hr_dip_pct": 14., "activity_day_mvpa_min_mean": 30.}
        count_fields = ["gait_n_valid_days", "sleep_diary_n_valid_nights", "hdcza_n_valid_nights",
                        "lower_back_n_valid_nights", "hrv_n_valid_nights", "hrv_rmssd_n_valid_nights",
                        "hr_n_valid_days", "hr_n_valid_nights", "hr_dip_n_valid_pairs", "activity_n_valid_days"]
        for folder in (self.bo, self.ra):
            for method in ("mean", "median"):
                self.csv(folder / f"overall_T0_sleep_{method}.csv", [
                    {"subject": str(subject), "visit": "T0", **measurements,
                     **{field: days for field in count_fields}}
                    for subject, days in ((7, 2), (8, 3))
                ])
        for method in ("mean", "median"):
            suffix = "" if method == "mean" else "_sleep_median"
            paths = write_combined_exports(output_dir=self.root / "output", sleep_method=method, **kwargs)
            for name in ("combined", "bologna", "ravenna"):
                self.assertEqual(paths[f"{name}_T0{suffix}"].name, f"fallspredict_{name}_T0{suffix}.csv")
                frame = pd.read_csv(paths[f"{name}_T0{suffix}"], dtype={"subject": "string"})
                self.assertTrue(frame.loc[frame.subject.eq("0007"), list(measurements)].isna().all().all())
                self.assertTrue(frame.loc[frame.subject.eq("0008"), list(measurements)].notna().all().all())
                self.assertTrue(frame.loc[frame.subject.eq("0007"), count_fields].eq(2).all().all())
                self.assertTrue(frame.has_sensors.eq(1).all())
            coverage = pd.read_csv(paths[f"column_coverage_T0{suffix}"]).set_index("variable")
            self.assertTrue(coverage.loc[list(measurements), ["BO_nonmissing_rows", "RA_nonmissing_rows"]].eq(1).all().all())


if __name__ == "__main__":
    unittest.main()
