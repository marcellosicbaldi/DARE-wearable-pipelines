"""Synthetic REDCap regression tests; no participant data is embedded here."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from fallspredict_gp_pipeline.aggregation.redcap import (
    compare_reference, load_redcap, merge_redcap_with_sensors,
    normalize_subjects, process_redcap, write_redcap_exports,
)
from fallspredict_gp_pipeline.redcap.schema import ATC_COLUMNS, DISEASE_COLUMNS, NON_SENSOR_COLUMNS
from fallspredict_gp_pipeline.redcap.scores import (
    CESD_REGULAR, CESD_REVERSED, FES_ITEMS, MMSE_ITEMS,
    PSQI_ITEMS, mmse_corrected_notebook, psqi_notebook, wfg_notebook,
)


def baseline(record="a", patient="7.0"):
    row = {
        "record_id": record, "patient_id": patient, "redcap_event_name": "baseline_arm_1",
        "patient_gender": "2", "patient_age": "75", "patient_school": "13",
        "patient_recruitment": "-2", "rec_oth": "volontario", "marital_status": "1",
        "encounter_fof_b": "2025-05-01", "dropout_yn": "", "dropout_r": "", "dropout_r_oth": "",
        "previous_fallsf_b": "0", "falls_lesionf_b": "", "falls_numf_b": "",
        "inability_to_get_upf_b": "", "temporary_consciousness_lossf_b": "",
        "dizziness_or_unsteadinessf_b": "0", "living_alonef_b": "1", "cfsf_b": "2",
        "walk_test_andata_b": "4", "walk_test_ritorno_b": "5", "test_timed_b": "",
        "daily_medications_spf_b": "2", "patologiesf_b": "0", "therapiesf_b": "0",
        "patologies_codef_b": "", "patologies_code_15f_b": "",
        "therapies_codef_b": "", "therapies_code_20f_b": "",
    }
    row.update({c: "1" for c in MMSE_ITEMS})
    row["best_scoref_b"] = "5"
    row.update({c: "1" for c in (*FES_ITEMS, *CESD_REGULAR)})
    row.update({c: "4" for c in CESD_REVERSED})
    row.update({f"physcial_disabilityf_b___{i}": "0" for i in range(1, 7)})
    row.update({f"instrumental_disabilityf_b___{i}": "1" for i in range(1, 9)})
    row.update({c: "0" for c in PSQI_ITEMS})
    row.update({"bed_timef_b": "23:00", "get_up_timef_b": "7.30", "hours_of_sleepf_b": "7",
                "falling_asleep_durationf_b": "20", "min_no_sleepf_b": "1"})
    return row


class RedcapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, rows):
        path = self.root / "source.csv"
        pd.DataFrame(rows).fillna("").to_csv(path, index=False)
        return path

    def test_record_scoped_identity_is_independent_of_order_and_preserves_raw(self):
        rows = [
            {"record_id": "b", "patient_id": "", "redcap_event_name": "mese_1_arm_1", "note": "NA"},
            baseline("a", "7"), baseline("b", "0012"),
            {"record_id": "a", "patient_id": "", "redcap_event_name": "mese_2_arm_1"},
        ]
        events = load_redcap(self.source(rows))
        self.assertEqual(events.subject.tolist(), ["0012", "0007", "0012", "0007"])
        self.assertEqual(events.loc[0, "patient_id"], "")
        self.assertEqual(events.loc[0, "note"], "NA")

    def test_ambiguous_and_missing_identity_fail(self):
        for rows in ([baseline("a", "7"), baseline("a", "8")],
                     [baseline("a", "7"), baseline("b", "7")],
                     [baseline("a", "")], [baseline("", "7")]):
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                load_redcap(self.source(rows))
        with self.assertRaises(ValueError):
            process_redcap(self.source([baseline(), baseline()]), fratup_dir=None)

    def test_subject_normalization_rejects_truncation_and_preserves_large_ids(self):
        self.assertEqual(normalize_subjects(pd.Series(["7.0", "0007", "22071"])).tolist(), ["0007", "0007", "22071"])
        for identifier in (None, "", "7.5", "nan", "-1"):
            with self.assertRaises(ValueError):
                normalize_subjects(pd.Series([identifier]))

    def test_clinical_schema_scores_and_missing_data(self):
        row = baseline()
        result = process_redcap(self.source([row]), fratup_dir=None)
        self.assertEqual(tuple(result.clinical), NON_SENSOR_COLUMNS)
        c = result.clinical.iloc[0]
        self.assertEqual(c["mmse"], 30)
        self.assertEqual(c["shortFESI"], 7)
        self.assertEqual(c["cesd"], 0)
        self.assertEqual(c["adl"], 0)
        self.assertEqual(c["iadl"], 8)
        self.assertEqual(c["gait_speed"], 0.8)
        self.assertEqual(c["history_falls_injurious"], 0)
        self.assertTrue(pd.isna(c["dropout_binario"]))
        self.assertTrue(pd.isna(c["fratup"]))
        self.assertAlmostEqual(c["mmse_corrected"], 2.4 * np.log10(93.9 - 75) * 13 ** 0.29 + 22.1)
        self.assertTrue(c[[*DISEASE_COLUMNS, *ATC_COLUMNS]].eq(0).all())
        row[MMSE_ITEMS[0]] = ""
        row[FES_ITEMS[0]] = ""
        row["walk_test_andata_b"] = "0"
        row["walk_test_ritorno_b"] = "-1"
        c = process_redcap(self.source([row]), fratup_dir=None).clinical.iloc[0]
        self.assertTrue(pd.isna(c["mmse"]))
        self.assertTrue(pd.isna(c["shortFESI"]))
        self.assertTrue(pd.isna(c["gait_speed"]))

    def test_absent_source_fields_are_reported(self):
        row = baseline()
        del row["patient_age"]
        result = process_redcap(self.source([row]), fratup_dir=None)
        self.assertTrue(pd.isna(result.clinical.age.iloc[0]))
        self.assertEqual(result.mapping.set_index("variable").loc["age", "status"], "missing_source_columns")

    def test_legacy_psqi_and_wfg_conventions_are_explicit(self):
        row = baseline()
        frame = pd.DataFrame([row])
        # Notebook converts wake 7.30 -> 07:50, yielding efficiency 79.2%.
        # C2=1, C3=1, C4=1, all other components=0.
        self.assertEqual(psqi_notebook(frame).iloc[0], 3)
        row.update({"previous_fallsf_b": "1", "cfsf_b": "2", "falls_numf_b": "1",
                    "walk_test_andata_b": "4", "walk_test_ritorno_b": "6",
                    "inability_to_get_upf_b": "sì"})
        self.assertEqual(wfg_notebook(pd.DataFrame([row])).iloc[0], "low")
        row["inability_to_get_upf_b"] = "SI"
        self.assertEqual(wfg_notebook(pd.DataFrame([row])).iloc[0], "high")
        row["previous_fallsf_b"] = ""
        self.assertEqual(wfg_notebook(pd.DataFrame([row])).iloc[0], "ND")
        for c in PSQI_ITEMS:
            row[c] = ""
        self.assertTrue(pd.isna(psqi_notebook(pd.DataFrame([row])).iloc[0]))

    def test_all_months_and_missing_counts_are_preserved(self):
        rows = [baseline(),
                {"record_id": "a", "redcap_event_name": "mese_2_arm_1", "cadute_mese": "1", "falls_encounter": "2", "cadute_mese_c2": "0"},
                {"record_id": "a", "redcap_event_name": "mese_6__follow_up_arm_1", "cadute_mese": "0"},
                {"record_id": "a", "redcap_event_name": "mese_12_arm_1", "cadute_mese": ""}]
        result = process_redcap(self.source(rows), fratup_dir=None)
        monthly = result.followup_monthly.set_index("month")
        self.assertEqual(monthly.index.tolist(), [2, 6, 12])
        self.assertEqual(monthly.loc[2, "n_falls_reported"], 2)
        self.assertEqual(monthly.loc[2, "fall_occurred"], 1)
        self.assertTrue(pd.isna(monthly.loc[6, "n_falls_reported"]))
        wide = result.followup_wide.iloc[0]
        self.assertEqual(wide["n_followup_months_observed"], 2)
        self.assertEqual(wide["followup_event_m12"], 1)
        self.assertEqual(wide["followup_event_m01"], 0)
        self.assertTrue(pd.isna(wide["fall_occurred_m01"]))
        self.assertEqual(wide["n_falls_reported_total"], 2)
        with self.assertRaises(ValueError):
            process_redcap(self.source(rows + [rows[1]]), fratup_dir=None)

    def test_codebook_checks_last_slots_and_does_not_turn_unknown_codes_into_zero(self):
        book = {"source": "Synthetic test definitions", "diseases": {c: [] for c in DISEASE_COLUMNS},
                "atc": {c: [] for c in ATC_COLUMNS}}
        book["diseases"][DISEASE_COLUMNS[0]] = [1]
        book["atc"][ATC_COLUMNS[0]] = [2]
        path = self.root / "codebook.json"
        path.write_text(json.dumps(book))
        a = baseline("a", "7")
        a.update({"patologies_code_15f_b": "1", "therapies_code_20f_b": "2"})
        b = baseline("b", "8")
        b.update({"patologies_codef_b": "9999", "patologies_code_15f_b": "1"})
        with warnings.catch_warnings(record=True) as caught:
            result = process_redcap(self.source([a, b]), codebook_path=path, fratup_dir=None).clinical
        self.assertTrue(caught)
        self.assertEqual(result.loc[0, "n_morbidities"], 1)
        self.assertEqual(result.loc[0, "n_medications"], 1)
        self.assertEqual(result.loc[1, DISEASE_COLUMNS[0]], 1)
        self.assertTrue(pd.isna(result.loc[1, DISEASE_COLUMNS[1]]))
        self.assertTrue(pd.isna(result.loc[1, "n_morbidities"]))
        path.write_text(json.dumps({"diseases": {}}))
        with self.assertRaises(ValueError):
            process_redcap(self.source([a]), codebook_path=path, fratup_dir=None)

    def test_merge_preserves_unmatched_subjects_and_prevents_row_multiplication(self):
        clinical = pd.DataFrame({"subject": ["0007", "0008"], "age": [75, 80]})
        sensors = pd.DataFrame({"subject": ["7.0", "9"], "visit": ["T0", "T0"], "sensor": [1, 2]})
        merged = merge_redcap_with_sensors(clinical, sensors)
        self.assertEqual(merged.subject.tolist(), ["0007", "0008", "0009"])
        self.assertEqual(merged.loc[0, "sensor"], 1)
        for bad in (pd.concat([sensors, sensors]), sensors.assign(age=50), sensors.assign(visit="T1")):
            with self.assertRaises(ValueError):
                merge_redcap_with_sensors(clinical, bad)

    def test_comparison_never_backfills_reference_values(self):
        result = process_redcap(self.source([baseline()]), fratup_dir=None).clinical
        reference = result.copy()
        reference["mmse_corrected"] = 99
        reference["guider_used"] = "sensor"
        path = self.root / "reference.csv"
        reference.to_csv(path, index=False)
        comparison = compare_reference(result, path).set_index("variable")
        self.assertEqual(comparison.loc["mmse_corrected", "different_nonmissing"], 1)
        self.assertTrue(result.mmse_corrected.ne(99).all())

    def test_new_notebook_codebook_and_overlap_audit(self):
        row = baseline()
        row.update({"patologies_codef_b": "313", "patologies_code_15f_b": "957",
                    "therapies_codef_b": "58", "therapies_code_20f_b": "101"})
        result = process_redcap(self.source([row]), fratup_dir=None)
        c = result.clinical.iloc[0]
        self.assertEqual(c.t0_Dyslipidemia, 1)
        self.assertEqual(c.t0_Dorsopathies, 0)
        self.assertEqual(c.t0_Solid_neoplasms, 1)
        self.assertEqual(c.t0_Venous_Lymp_dis, 1)
        self.assertEqual(c.n_morbidities, 3)
        self.assertEqual(c.J01, 1)
        self.assertEqual(c.V20, 1)
        self.assertEqual(c.n_medications, 2)
        issues = result.codebook_issues
        self.assertEqual(issues.issue.tolist(), ["identical_category_codes"])

    def test_mmse_equation_domain_and_snapshot_identity_validation(self):
        result = mmse_corrected_notebook(pd.Series([75, 93.9, 100, None]), pd.Series([13, 13, 13, 13]))
        self.assertTrue(result.iloc[1:].isna().all())
        source = self.source([baseline()])
        snapshot = load_redcap(source)
        snapshot.to_csv(source, index=False)
        pd.testing.assert_frame_equal(snapshot, load_redcap(source))
        snapshot["subject"] = "9999"
        snapshot.to_csv(source, index=False)
        with self.assertRaises(ValueError):
            load_redcap(source)

    def test_output_contains_every_original_event_field_and_checks_input_collision(self):
        source = self.source([baseline()])
        paths = write_redcap_exports(source, output_dir=self.root / "out", fratup_dir=None)
        raw = pd.read_csv(source, dtype="string", keep_default_na=False)
        exported = pd.read_csv(paths["events"], dtype="string", keep_default_na=False)
        pd.testing.assert_frame_equal(raw, exported.drop(columns="subject"))
        collision = self.root / "redcap_events.csv"
        raw.to_csv(collision, index=False)
        with self.assertRaises(ValueError):
            write_redcap_exports(collision, output_dir=self.root, fratup_dir=None)
        pd.testing.assert_frame_equal(raw, pd.read_csv(collision, dtype="string", keep_default_na=False))

    def test_overall_integration_and_sensor_only_behavior(self):
        from fallspredict_gp_pipeline.aggregation.overall import aggregate_all, write_all_exports
        source = self.source([baseline()])
        sensors = pd.DataFrame({"subject": ["7"], "visit": ["T0"], "sleep_value": [10]})
        with patch("fallspredict_gp_pipeline.aggregation.overall.aggregate_sleep", return_value=sensors), \
             patch("fallspredict_gp_pipeline.aggregation.overall._build_non_sleep_domain_frames", return_value=[]), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.testing.assert_frame_equal(aggregate_all(), sensors)
            merged = aggregate_all(redcap_csv=source, fratup_dir=None)
            self.assertEqual(len(merged), 1)
            self.assertEqual(merged.subject.iloc[0], "0007")
            self.assertEqual(merged.mmse.iloc[0], 30)
            outputs = write_all_exports(output_dir=self.root / "all", redcap_csv=source, fratup_dir=None)
            self.assertEqual(set(outputs), {"mean", "median"})
            self.assertTrue(all(path.exists() for path, _ in outputs.values()))
            with self.assertRaises(ValueError):
                aggregate_all(redcap_csv=source, visit="T1", fratup_dir=None)


if __name__ == "__main__":
    unittest.main()
