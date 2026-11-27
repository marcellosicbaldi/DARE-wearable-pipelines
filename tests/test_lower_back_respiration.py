"""Synthetic checks for the experimental spectral screen and nonwear override.

These establish implementation behavior, not validity on participant data.
"""
from io import StringIO
import unittest

import numpy as np
import pandas as pd

from scripts.lower_back_respiration import AXES, analyze_gyro, apply_override


class LowerBackRespirationTests(unittest.TestCase):
    fs = 100.0

    def gyro(self, seconds=120, tone_axis="gy", frequency=.25, amplitude=.06):
        t = np.arange(round(seconds * self.fs)) / self.fs
        frame = pd.DataFrame(0., index=pd.date_range("2024-01-01", periods=len(t), freq="10ms"),
                             columns=list(AXES))
        if tone_axis is not None:
            frame[tone_axis] = amplitude * np.sin(2 * np.pi * frequency * t)
        return frame

    def baseline(self, *intervals):
        return pd.DataFrame({
            "start_sample": [round(first * self.fs) for first, _ in intervals],
            "stop_sample": [round(stop * self.fs) for _, stop in intervals],
        })

    def test_one_axis_tone_is_sufficient_at_both_window_lengths(self):
        frame = self.gyro()
        for seconds in (30, 60):
            with self.subTest(seconds=seconds):
                result = analyze_gyro(frame, self.fs, window_seconds=seconds)
                self.assertTrue(result.windows.respiration_present.all())
                self.assertTrue(result.windows.gy_positive.all())
                self.assertFalse(result.windows.gx_positive.any())
                self.assertFalse(result.windows.gz_positive.any())
                np.testing.assert_allclose(result.windows.gy_peak_hz, .25, atol=1 / seconds)
                np.testing.assert_array_equal(np.diff(result.windows.start_sample), seconds * self.fs / 2)

    def test_flat_tiny_and_out_of_band_signals_do_not_pass(self):
        cases = (self.gyro(tone_axis=None), self.gyro(amplitude=.0001),
                 self.gyro(frequency=1.3))
        for frame in cases:
            result = analyze_gyro(frame, self.fs, window_seconds=60)
            self.assertFalse(result.windows.respiration_present.any())

    def test_nonfinite_axis_does_not_hide_a_valid_positive_axis(self):
        frame = self.gyro()
        frame["gx"] = np.nan
        result = analyze_gyro(frame, self.fs, window_seconds=60)
        self.assertTrue(result.windows.respiration_present.all())
        self.assertFalse(result.windows.gx_valid.any())
        self.assertTrue(np.isnan(result.psd[:, 0]).all())

    def test_only_overlapping_nonwear_is_rescued(self):
        frame = self.gyro()
        result = analyze_gyro(frame, self.fs, window_seconds=30)
        result.windows["respiration_present"] = False
        result.windows.loc[0, "respiration_present"] = True  # evidence at [0, 30 s)
        baseline = self.baseline((20, 50), (80, 90))
        changed = apply_override(result, baseline, frame.index)
        self.assertEqual(changed["summary"]["reassigned_to_wear_hours"] * 3600, 10.)
        self.assertEqual(changed["nonwear_bouts"].start_sample.tolist(), [3000, 8000])
        self.assertEqual(changed["nonwear_bouts"].stop_sample.tolist(), [5000, 9000])
        self.assertEqual(changed["rescued_bouts"].start_sample.tolist(), [2000])
        self.assertEqual(changed["rescued_bouts"].stop_sample.tolist(), [3000])

    def test_overlapping_positive_windows_count_once(self):
        frame = self.gyro()
        result = analyze_gyro(frame, self.fs, window_seconds=30)
        result.windows["respiration_present"] = False
        result.windows.loc[[0, 1], "respiration_present"] = True  # union [0, 45 s)
        changed = apply_override(result, self.baseline((0, 60)), frame.index)
        self.assertEqual(changed["summary"]["reassigned_to_wear_hours"] * 3600, 45.)
        self.assertEqual(changed["summary"]["revised_nonwear_hours"] * 3600, 15.)

    def test_respiration_never_creates_nonwear(self):
        frame = self.gyro()
        result = analyze_gyro(frame, self.fs, window_seconds=60)
        # Reading an empty saved baseline yields object-typed columns.
        empty_baseline = pd.read_csv(StringIO("start_sample,stop_sample,start,end\n"))
        changed = apply_override(result, empty_baseline, frame.index)
        self.assertTrue(changed["nonwear_bouts"].empty)
        self.assertTrue(changed["rescued_bouts"].empty)
        self.assertEqual(changed["summary"]["reassigned_to_wear_hours"], 0.)

    def test_unassessed_nonwear_is_preserved(self):
        frame = self.gyro()
        frame.loc[:, :] = np.nan
        result = analyze_gyro(frame, self.fs, window_seconds=60)
        changed = apply_override(result, self.baseline((0, 120)), frame.index)
        self.assertEqual(changed["summary"]["baseline_nonwear_assessed_hours"], 0.)
        self.assertTrue(changed["rescued_bouts"].empty)
        self.assertEqual(changed["nonwear_bouts"].duration_seconds.sum(), 120.)

    def test_incomplete_tail_is_unassessed_and_keeps_existing_nonwear(self):
        frame = self.gyro(seconds=125)
        result = analyze_gyro(frame, self.fs, window_seconds=30)
        changed = apply_override(result, self.baseline((0, 125)), frame.index)
        self.assertEqual(changed["nonwear_bouts"].start_sample.tolist(), [12000])
        self.assertEqual(changed["nonwear_bouts"].stop_sample.tolist(), [12500])
        self.assertAlmostEqual(changed["summary"]["baseline_nonwear_unassessed_hours"] * 3600, 5.)

    def test_windows_and_exported_bouts_do_not_bridge_timestamp_gaps(self):
        frame = self.gyro()
        values = frame.index.to_numpy(copy=True)
        values[6000:] += np.timedelta64(60, "s")
        frame.index = pd.DatetimeIndex(values)
        result = analyze_gyro(frame, self.fs, window_seconds=30)
        self.assertFalse(((result.windows.start_sample < 6000) &
                          (result.windows.stop_sample > 6000)).any())
        changed = apply_override(result, self.baseline((0, 120)), frame.index)
        rescued = changed["rescued_bouts"]
        self.assertEqual(len(rescued), 2)
        self.assertEqual(rescued.duration_seconds.sum(), 120.)
        self.assertEqual((rescued.start.iloc[1] - rescued.end.iloc[0]).total_seconds(), 60.)


if __name__ == "__main__":
    unittest.main()
