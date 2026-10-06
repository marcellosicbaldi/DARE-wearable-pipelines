"""Synthetic measurements testing the user-specified analysis exclusion rules."""

import unittest

import pandas as pd

from fallspredict_gp_pipeline.aggregation.analysis_exclusions import (
    EXCLUDED_ANALYSIS_COLUMNS, AnalysisExclusions,
    apply_analysis_exclusions,
)


SYNTHETIC_HRV_IDS = ("910001", "910002")
SYNTHETIC_SLEEP_IDS = ("920001", "920002")


class AnalysisExclusionTests(unittest.TestCase):
    def setUp(self):
        self.exclusions = AnalysisExclusions(SYNTHETIC_HRV_IDS, SYNTHETIC_SLEEP_IDS)

    def test_fewer_than_three_gait_days_mask_both_cohorts_but_preserve_qc_and_other_domains(self):
        frame = pd.DataFrame({
            "group": ["BO", "RA", "BO", "RA", "BO", "RA"],
            "subject": ["9999"] * 6, "visit": ["T0"] * 6,
            "gait_n_valid_days": pd.Series([0, "1.0", 2, 3, 4, 5], dtype="string"),
            "gait_n_days": [5] * 6, "gait_mean_valid_hours": [10., 12., 17., 18., None, None],
            "gait_step_count": [2000.] * 6,
            "gait_wb_30__walking_speed_mps__avg": [0.8] * 6,
            "gait_wb_all__cadence_spm__avg": [90.] * 6,
            "gait_wb_all__stride_duration_s__var": [0.1] * 6,
            "hrv_rmssd": [30.] * 6, "sleep_diary_tst_min": [420.] * 6,
            "hrv_n_valid_nights": [3] * 6, "hrv_rmssd_n_valid_nights": [3] * 6,
            "sleep_diary_n_valid_nights": [3] * 6,
            "age": [75.] * 6, "has_sensors": [1] * 6,
        }, index=range(6))
        # A non-default index checks that masking follows rows, not positions.
        frame.index = [10, 20, 30, 40, 50, 60]
        before = frame.copy(deep=True)
        result = apply_analysis_exclusions(frame, exclusions=self.exclusions)
        features = ["gait_step_count", "gait_wb_30__walking_speed_mps__avg",
                    "gait_wb_all__cadence_spm__avg", "gait_wb_all__stride_duration_s__var"]
        self.assertTrue(result.loc[[10, 20, 30], features].isna().all().all())
        pd.testing.assert_frame_equal(result.loc[[40, 50, 60], features], frame.loc[[40, 50, 60], features])
        for column in set(frame).difference(features):
            pd.testing.assert_series_equal(result[column], frame[column])
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(result, apply_analysis_exclusions(result, exclusions=self.exclusions))
        # Legacy populated exports without counts must be refreshed.
        with self.assertRaisesRegex(ValueError, "gait_n_valid_days"):
            apply_analysis_exclusions(frame.drop(columns="gait_n_valid_days"), exclusions=self.exclusions)

    def test_synthetic_hrv_ids_with_padding_only_target_bo_t0(self):
        ids = [int(i) for i in SYNTHETIC_HRV_IDS] + ["0910001", "00910001.0", "910001", "910001", "9999"]
        count = len(ids)
        excluded_count = len(SYNTHETIC_HRV_IDS) + 2
        frame = pd.DataFrame({"group": ["BO"] * excluded_count + ["RA", "BO", "BO"], "subject": ids,
                              "visit": ["T0"] * (excluded_count + 1) + ["T1", "T0"],
                              "hrv_rmssd": [30.] * count, "hrv_sdnn": [40.] * count,
                              "hrv_mean_hr": [65.] * count, "hrv_PIP": [0.3] * count,
                              "sleep_diary_tst_min": [420.] * count, "gait_step_count": [2000] * count,
                              "sleep_diary_n_valid_nights": [3] * count, "gait_n_valid_days": [3] * count,
                              "hrv_n_valid_nights": [3] * count,
                              **{f"hrv_{metric}_n_valid_nights": [3] * count for metric in ("rmssd", "sdnn", "mean_hr", "PIP")}})
        result = apply_analysis_exclusions(frame, exclusions=self.exclusions)
        self.assertTrue(result.iloc[:excluded_count][["hrv_rmssd", "hrv_sdnn"]].isna().all().all())
        self.assertEqual(result.rmssd_sdnn_exclusion.tolist(), [1] * excluded_count + [0] * 3)
        self.assertTrue(result.rmssd_sdnn_exclusion_method.iloc[:excluded_count].eq("visual inspection").all())
        self.assertTrue(result.rmssd_sdnn_exclusion_method.iloc[excluded_count:].isna().all())
        self.assertTrue(result.hrv_rmssd.iloc[excluded_count:].eq(30).all())
        for field in ("hrv_mean_hr", "hrv_PIP", "sleep_diary_tst_min", "gait_step_count"):
            pd.testing.assert_series_equal(result[field], frame[field])

    def test_sleep_ids_mask_all_sensor_domains_and_quality_counts_only_in_ra_t0(self):
        ids = list(SYNTHETIC_SLEEP_IDS) + [920001, "0920002.0", "0920001", "0920001", "99999"]
        count = len(ids)
        excluded_count = len(SYNTHETIC_SLEEP_IDS) + 2
        frame = pd.DataFrame({"group": ["RA"] * excluded_count + ["BO", "RA", "RA"], "subject": ids,
                              "visit": ["T0"] * (excluded_count + 1) + ["T1", "T0"],
                              "sleep_diary_tst_min": [420.] * count, "sleep_diary_n_valid_nights": [4] * count,
                              "hdcza_waso_min": [30.] * count, "hdcza_n_valid_nights": [5] * count,
                              "lower_back_sol_min": [10.] * count, "lower_back_n_valid_nights": [4] * count,
                              "circadian_IS": [0.7] * count, "circadian_L5TIME_clock": ["01:30"] * count,
                              "activity_day_mvpa_min_mean": [40.] * count, "activity_n_valid_days": [5] * count,
                              "gait_step_count": [2000] * count, "hrv_rmssd": [30.] * count,
                              "gait_n_valid_days": [3] * count, "hrv_n_valid_nights": [3] * count,
                              "hrv_rmssd_n_valid_nights": [3] * count,
                              "psqi": [4.] * count, "age": [75.] * count, "has_sensors": [1] * count,
                              "fratup": [0.2] * count})
        before = frame.copy(deep=True)
        result = apply_analysis_exclusions(frame, exclusions=self.exclusions)
        fields = [c for c in frame if c.startswith(("sleep_diary_", "hdcza_", "lower_back_", "circadian_", "activity_"))]
        self.assertTrue(result.iloc[:excluded_count][fields].isna().all().all())
        pd.testing.assert_frame_equal(result.iloc[excluded_count:][fields], frame.iloc[excluded_count:][fields], check_dtype=False)
        self.assertEqual(result.sleep_circadian_exclusion.tolist(), [1] * excluded_count + [0] * 3)
        self.assertTrue(result.sleep_circadian_exclusion_method.iloc[:excluded_count].eq("visual inspection").all())
        self.assertTrue(result.sleep_circadian_exclusion_method.iloc[excluded_count:].isna().all())
        for field in ("gait_step_count", "hrv_rmssd", "psqi", "age", "has_sensors", "fratup"):
            pd.testing.assert_series_equal(result[field], frame[field])
        pd.testing.assert_frame_equal(frame, before)

    def test_exact_drop_list_and_similarly_named_columns_are_preserved(self):
        requested = {
            "hdcza_spt_start_clock_h", "hdcza_spt_end_clock_h", "hdcza_guider_start_clock_h", "hdcza_guider_end_clock_h",
            "gait_wb_all__n_raw_initial_contacts__sum", "gait_wb_all__n_turns__sum",
            "hrv_window_length_s", "hrv_source_night_id", "hrv_priority_rank",
            "hrv_PIP_percent_discarded_median", "hrv_mean_hr_percent_discarded_median",
            "hrv_rmssd_percent_discarded_median", "hrv_sdnn_percent_discarded_median",
            "circadian_MESOR_log1p_mg", "circadian_Amplitude_log1p_mg", "circadian_Acrotime_hour", "circadian_Cosinor_n_days_used",
        }
        self.assertEqual(set(EXCLUDED_ANALYSIS_COLUMNS), requested)
        frame = pd.DataFrame({"group": ["BO"], "subject": ["9999"], "visit": ["T0"],
                              **{field: [1.] for field in requested}, "circadian_Cosinor_n_valid_minutes": [1440.],
                              "circadian_Phase_rad_series_start": [0.3], "circadian_cosinor_acrotime": [12.],
                              "hrv_quiet_segment_length_s": [300.], "hrv_n_valid_nights": [3]})
        result = apply_analysis_exclusions(frame, exclusions=self.exclusions)
        self.assertFalse(requested.intersection(result.columns))
        for field in set(frame).difference(requested):
            pd.testing.assert_series_equal(result[field], frame[field])

    def test_missing_measurements_still_record_the_decision_and_repeat_is_idempotent(self):
        frame = pd.DataFrame({"group": ["BO", "RA", "BO"], "subject": ["910001", "920001", "9999"],
                              "visit": ["T0"] * 3, "hrv_rmssd": [float("nan")] * 3,
                              "sleep_diary_tst_min": [float("nan")] * 3})
        result = apply_analysis_exclusions(frame, exclusions=self.exclusions)
        self.assertEqual(result.rmssd_sdnn_exclusion.tolist(), [1, 0, 0])
        self.assertEqual(result.sleep_circadian_exclusion.tolist(), [0, 1, 0])
        pd.testing.assert_frame_equal(result, apply_analysis_exclusions(result, exclusions=self.exclusions))
        self.assertEqual(len(result), len(frame))


if __name__ == "__main__":
    unittest.main()
