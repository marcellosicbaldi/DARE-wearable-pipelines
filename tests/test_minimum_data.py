"""Synthetic coverage checks: independent domains, filtered nights and boundaries."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gp_pipeline.aggregation.heart_rate import aggregate_heart_rate
from gp_pipeline.aggregation.hrv import aggregate_hrv_windows
from gp_pipeline.aggregation.minimum_data import apply_minimum_observations


class MinimumDataTests(unittest.TestCase):
    def test_each_domain_uses_its_own_count_and_keeps_three(self):
        values = {
            "gait_n_valid_days": 2, "gait_step_count": 3000., "gait_n_days": 7,
            "sleep_diary_n_valid_nights": 2, "sleep_diary_tst_min": 420.,
            "hdcza_n_valid_nights": 3, "hdcza_tst_min": 410.,
            "lower_back_n_valid_nights": 1, "lower_back_tst_min": 400.,
            "hrv_n_valid_nights": 4, "hrv_n_windows": 30,
            "hrv_rmssd_n_valid_nights": 2, "hrv_rmssd": 25.,
            "hrv_sdnn_n_valid_nights": 3, "hrv_sdnn": 40.,
            "hr_n_valid_days": 2, "median_hr_day": 75.,
            "hr_n_valid_nights": 3, "median_hr_night": 60.,
            "hr_dip_n_valid_pairs": 1, "hr_dip_pct": 20.,
            "activity_n_valid_days": 2, "activity_day_mvpa_min_mean": 30.,
            "age": 70., "fratup": 0.2, "gait_speed": 1.1, "has_sensors": 1,
        }
        frame = pd.DataFrame([values], index=[42])
        before = frame.copy(deep=True)
        result = apply_minimum_observations(frame)
        excluded = ["gait_step_count", "sleep_diary_tst_min", "lower_back_tst_min",
                    "hrv_rmssd", "median_hr_day", "hr_dip_pct", "activity_day_mvpa_min_mean"]
        self.assertTrue(result[excluded].isna().all().all())
        for column in set(frame).difference(excluded):
            pd.testing.assert_series_equal(result[column], frame[column])
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(result, apply_minimum_observations(result))

    def test_missing_counts_require_refresh_only_when_measurements_are_populated(self):
        for count in (None, float("nan"), "bad"):
            frame = pd.DataFrame({"median_hr_day": [70.], "hr_n_valid_days": [count]})
            with self.assertRaisesRegex(ValueError, "hr_n_valid_days"):
                apply_minimum_observations(frame)
        empty = pd.DataFrame({"median_hr_day": [np.nan], "hrv_rmssd": [np.nan], "gait_step_count": [np.nan]})
        pd.testing.assert_frame_equal(empty, apply_minimum_observations(empty))
        clinical_only = pd.DataFrame({"gait_speed": [1.1]})
        pd.testing.assert_frame_equal(clinical_only, apply_minimum_observations(clinical_only))
        with self.assertRaisesRegex(ValueError, "hrv_rmssd_n_valid_nights"):
            apply_minimum_observations(pd.DataFrame({"hrv_rmssd": [30.], "hrv_n_valid_nights": [5]}))

    def test_hr_counts_nonmissing_distinct_nights_and_day_night_pairs_separately(self):
        nightly = pd.DataFrame({
            "subject": ["7"] * 6, "visit": ["T0"] * 6, "night_id": [1, 1, 2, 3, 4, 5],
            "median_hr_day": [70., 70., 72., np.nan, np.nan, np.nan],
            "median_hr_night": [60., 60., np.nan, 62., 64., np.nan],
            "hr_dip_pct": [14., 14., np.nan, 12., np.nan, np.nan],
        })
        with patch("gp_pipeline.aggregation.heart_rate._discover_heart_rate_inputs", return_value=[("7", Path("hr"), Path("sleep"))]), \
             patch("gp_pipeline.aggregation.heart_rate.aggregate_participant_heart_rate", return_value=nightly):
            result, audit = aggregate_heart_rate("unused", recruitment_tracker_path=None)
        row = result.iloc[0]
        self.assertEqual(row.hr_n_valid_days, 2)
        self.assertEqual(row.hr_n_valid_nights, 3)
        self.assertEqual(row.hr_dip_n_valid_pairs, 2)
        self.assertTrue(pd.isna(row.median_hr_day))
        self.assertEqual(row.median_hr_night, 62.)
        self.assertTrue(pd.isna(row.hr_dip_pct))
        pd.testing.assert_frame_equal(audit, nightly)

    def test_hrv_counts_after_filtering_per_metric_and_does_not_count_windows_as_nights(self):
        windows = pd.DataFrame({
            "subject": ["7"] * 6, "day": [1, 1, 2, 3, 4, 5],
            "rmssd": [10., 20., np.nan, 30., np.nan, np.nan],
            "sdnn": [30., 50., 60., 70., np.nan, np.nan],
            "mean_hr": [60., 60., 61., 62., 63., np.nan],
            "PIP": [0.2, 0.2, 0.3, np.nan, np.nan, np.nan],
            "n_beats": [200] * 6, "rmssd_outlier": [False] * 6,
        })
        row = aggregate_hrv_windows(windows).iloc[0]
        self.assertEqual(row.hrv_n_valid_nights, 4)
        self.assertEqual(row.hrv_n_windows, 6)
        self.assertEqual(row.hrv_rmssd_n_valid_nights, 2)
        self.assertEqual(row.hrv_sdnn_n_valid_nights, 3)
        self.assertEqual(row.hrv_mean_hr_n_valid_nights, 4)
        self.assertTrue(pd.isna(row.hrv_rmssd))
        self.assertTrue(pd.isna(row.hrv_PIP))
        self.assertEqual(row.hrv_sdnn, 60.)
        self.assertEqual(row.hrv_mean_hr, 61.5)
        invalid = windows.copy()
        invalid[["rmssd", "sdnn", "mean_hr", "PIP"]] = np.nan
        row = aggregate_hrv_windows(invalid).iloc[0]
        self.assertEqual(row.hrv_n_valid_nights, 0)
        self.assertTrue(pd.isna(row.hrv_n_beats))


if __name__ == "__main__":
    unittest.main()
