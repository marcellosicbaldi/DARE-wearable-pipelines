"""Synthetic timestamp-boundary regressions for the exploratory comparison."""
import unittest

import numpy as np
import pandas as pd

from scripts.compare_lower_back_nonwear import (
    FS,
    _active_summary,
    _mask,
    _production_mask_from_windows,
    _python_windows,
)


class ComparisonMethodTests(unittest.TestCase):
    def test_old_default_threshold_run_cannot_be_used_as_three_axis_results(self):
        summary = pd.DataFrame({"method": ["current", "ggir_2023_matched", "ggir_2023_defaults"]})
        with self.assertRaisesRegex(ValueError, "ggir_2023_matched_3axes"):
            _active_summary(summary)

    def test_retired_default_threshold_rows_are_excluded(self):
        summary = pd.DataFrame({
            "method": ["current", "ggir_2023_matched", "ggir_2023_matched_3axes", "ggir_2023_defaults"],
            "detected_nonwear_hours": [1., 2., .5, 9.],
        })
        active = _active_summary(summary)
        self.assertEqual(active.method.tolist(), ["current", "ggir_2023_matched", "ggir_2023_matched_3axes"])
        self.assertEqual(active.detected_nonwear_hours.tolist(), [1., 2., .5])


class NonwearComparisonTests(unittest.TestCase):
    def make_recording(self, offset_ns=999, flags=(True, True, False, True), tail=7):
        epoch = int(15 * 60 * FS)
        times = pd.date_range(
            pd.Timestamp("2024-01-01") + pd.Timedelta(offset_ns, unit="ns"),
            periods=len(flags) * epoch + tail, freq="10ms",
        )
        starts = np.arange(len(flags)) * epoch
        stops = starts + epoch
        windows = pd.DataFrame({
            "win_start": times[starts].as_unit("us"),
            "win_end": times[stops - 1].as_unit("us"),
            "is_nonwear_window": flags,
            "is_final_nonwear": flags,
        })
        return times, windows, epoch

    def test_epoch_bounds_keep_samples_with_nanosecond_offsets(self):
        times, exported, epoch = self.make_recording()
        windows = _python_windows(exported, times)
        self.assertEqual(windows.start_sample.tolist(), [0, epoch, 2 * epoch, 3 * epoch])
        self.assertEqual(windows.stop_sample.tolist(), [epoch, 2 * epoch, 3 * epoch, 4 * epoch])
        mask = _mask(windows, "is_final_nonwear", len(times))
        self.assertTrue(mask[epoch - 1])  # no artificial hole between adjacent epochs
        self.assertFalse(mask[-1])  # final partial epoch stays outside the decisions

    def test_production_mask_rounds_only_merged_bout_ends(self):
        times, exported, epoch = self.make_recording()
        windows = _python_windows(exported, times)
        actual = _production_mask_from_windows(windows, times)
        expected = np.zeros(len(times), dtype=bool)
        expected[:2 * epoch - 1] = True
        expected[3 * epoch:4 * epoch - 1] = True
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(actual[epoch - 1])
        self.assertFalse(actual[2 * epoch - 1])
        epoch_mask = _mask(windows, "is_final_nonwear", len(times))
        self.assertEqual(np.count_nonzero(actual ^ epoch_mask), 2)

    def test_microsecond_aligned_samples_have_no_rounding_difference(self):
        times, exported, _ = self.make_recording(offset_ns=0)
        windows = _python_windows(exported, times)
        np.testing.assert_array_equal(
            _production_mask_from_windows(windows, times),
            _mask(windows, "is_final_nonwear", len(times)),
        )

    def test_all_wear_is_preserved(self):
        times, exported, _ = self.make_recording(flags=(False, False))
        windows = _python_windows(exported, times)
        self.assertFalse(_production_mask_from_windows(windows, times).any())

    def test_timestamp_gap_does_not_change_sample_defined_epoch_bounds(self):
        times, exported, epoch = self.make_recording()
        values = times.to_numpy(copy=True)
        values[epoch // 2:] += np.timedelta64(2, "s")
        times = pd.DatetimeIndex(values)
        starts = np.arange(len(exported)) * epoch
        exported["win_start"] = times[starts].as_unit("us")
        exported["win_end"] = times[starts + epoch - 1].as_unit("us")
        windows = _python_windows(exported, times)
        self.assertEqual(windows.stop_sample.tolist(), [epoch, 2 * epoch, 3 * epoch, 4 * epoch])
        self.assertTrue(_production_mask_from_windows(windows, times)[epoch - 1])

    def test_timestamp_disagreement_is_still_rejected(self):
        times, exported, _ = self.make_recording()
        exported.loc[1, "win_end"] += pd.Timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError, "timestamps do not match"):
            _python_windows(exported, times)


if __name__ == "__main__":
    unittest.main()
