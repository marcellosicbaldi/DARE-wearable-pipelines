"""Binary mounting states, conservative evidence and shared posture contracts."""
import unittest

import numpy as np
import pandas as pd

from dare_wearables.lower_back.orientation import (
    ACC, GYR, OrientationOptions, annotate_tib_orientation, correct_belt_inversion,
)
from dare_wearables.lower_back.posture.lying_functions import detect_lying_lowerback


def walking(data, fs):
    return pd.DataFrame({"start": [0, 30*fs, 60*fs], "end": [20*fs, 50*fs, 80*fs]})


class BeltInversionTests(unittest.TestCase):
    def frame(self, gravity=(9.80665, 0., 0.), seconds=120):
        index = pd.date_range("2026-01-01", periods=seconds*100, freq="10ms", name="datetime")
        data = pd.DataFrame(np.tile(gravity, (len(index), 1)), index=index, columns=ACC)
        data[GYR] = [5., 7., 9.]
        data["wear"] = True
        return data

    def test_only_expected_axes_reverse_and_tilt_is_preserved(self):
        original = self.frame((8., 1., 4.))
        mounted = original.copy()
        mounted[["acc_x", "acc_y", "gyr_x", "gyr_y"]] *= -1
        before = mounted.copy()
        corrected, episodes, _, summary = correct_belt_inversion(mounted, walking_detector=walking)
        pd.testing.assert_frame_equal(corrected, original)
        pd.testing.assert_frame_equal(mounted, before)
        self.assertEqual(episodes.status.tolist(), ["inverted"])
        self.assertFalse(summary["tilt_alignment"])
        self.assertTrue(summary["wear_mask_unchanged"])
        self.assertTrue(corrected.acc_z.eq(4.).all())
        self.assertTrue(corrected.gyr_z.eq(9.).all())

    def test_upright_tilt_is_untouched(self):
        original = self.frame((8., 1., 4.))
        corrected, episodes, _, _ = correct_belt_inversion(original, walking_detector=walking)
        pd.testing.assert_frame_equal(corrected, original)
        self.assertEqual(episodes.status.tolist(), ["upright"])

    def test_real_walking_detection_and_correction_preserve_sequence_boundaries(self):
        from dare_wearables.lower_back.orientation import _walking_sequences
        original = self.frame(seconds=180)
        t = np.arange(len(original))/100
        for a, b in [(10, 40), (70, 100), (130, 160)]:
            selected = (t >= a) & (t < b)
            original.loc[selected, "acc_x"] += 2*np.sin(4*np.pi*t[selected])
        original["acc_y"] = .1*np.sin(2*np.pi*t)
        original["acc_z"] = .1*np.cos(4*np.pi*t)
        mounted = original.copy()
        mounted[["acc_x", "acc_y", "gyr_x", "gyr_y"]] *= -1
        before = _walking_sequences(original, 100)
        self.assertGreaterEqual(len(before), 3)
        corrected, episodes, _, _ = correct_belt_inversion(mounted)
        self.assertEqual(episodes.status.tolist(), ["inverted"])
        pd.testing.assert_frame_equal(corrected, original)
        pd.testing.assert_frame_equal(_walking_sequences(corrected, 100), before)

    def test_horizontal_lying_cannot_supply_an_inversion_vote(self):
        original = self.frame((0., -9.80665, 0.))
        corrected, episodes, evidence, _ = correct_belt_inversion(original, walking_detector=walking)
        pd.testing.assert_frame_equal(corrected, original)
        self.assertEqual(episodes.status.tolist(), ["insufficient_walking_evidence"])
        self.assertTrue(evidence.vote.eq("unreliable").all())

    def test_conflicting_states_are_flagged_instead_of_majority_corrected(self):
        original = self.frame(seconds=240)
        original.iloc[12000:, original.columns.get_indexer(["acc_x", "acc_y", "gyr_x", "gyr_y"])] *= -1
        def six_sequences(data, fs):
            return pd.DataFrame({"start": np.array([0, 30, 60, 120, 150, 180])*fs,
                                 "end": np.array([20, 50, 80, 140, 170, 200])*fs})
        corrected, episodes, _, _ = correct_belt_inversion(original, walking_detector=six_sequences)
        pd.testing.assert_frame_equal(corrected, original)
        self.assertEqual(episodes.status.tolist(), ["conflicting_walking_evidence"])

    def test_removal_and_timestamp_segments_allow_independent_states(self):
        for boundary in ("nonwear", "gap", "time_segment"):
            original = self.frame(seconds=270)
            original.iloc[15000:, original.columns.get_indexer(["acc_x", "acc_y", "gyr_x", "gyr_y"])] *= -1
            if boundary == "nonwear":
                original.iloc[12000:15000, original.columns.get_loc("wear")] = False
            else:
                original = pd.concat([original.iloc[:12000], original.iloc[15000:]])
                if boundary == "time_segment":
                    original.index = pd.date_range("2026-01-01", periods=len(original), freq="10ms")
                    original["time_segment"] = np.repeat([0, 1], 12000)
                original.index = original.index.as_unit("us")
            seen = []
            def detector(data, fs):
                self.assertTrue(data.wear.all())
                seen.append(len(data))
                return walking(data, fs)
            with self.subTest(boundary=boundary):
                corrected, episodes, _, _ = correct_belt_inversion(original, walking_detector=detector)
                self.assertEqual(seen, [12000, 12000])
                self.assertEqual(episodes.status.tolist(), ["upright", "inverted"])
                self.assertTrue(corrected.loc[corrected.wear, "acc_x"].gt(0).all())
                pd.testing.assert_series_equal(corrected.wear, original.wear)

    def test_insufficient_and_disabled_evidence_never_guess(self):
        data = self.frame((-9.80665, 0., 0.))
        no_sequences = lambda data, fs: pd.DataFrame(columns=["start", "end"])
        corrected, episodes, _, _ = correct_belt_inversion(data, walking_detector=no_sequences)
        pd.testing.assert_frame_equal(corrected, data)
        self.assertEqual(episodes.status.tolist(), ["insufficient_walking_evidence"])
        def forbidden(*args):
            raise AssertionError("Disabled detection was called")
        corrected, episodes, _, _ = correct_belt_inversion(data, options=OrientationOptions(method="none"), walking_detector=forbidden)
        pd.testing.assert_frame_equal(corrected, data)
        self.assertEqual(episodes.status.tolist(), ["disabled"])

    def test_correcting_inversion_preserves_real_lying(self):
        original = self.frame(seconds=900)
        original.loc[original.index[30000:60000], ACC] = [0., 9.80665, 0.]
        mounted = original.copy()
        mounted[["acc_x", "acc_y", "gyr_x", "gyr_y"]] *= -1
        corrected, _, _, _ = correct_belt_inversion(mounted, walking_detector=walking)
        before = detect_lying_lowerback(original)
        after = detect_lying_lowerback(corrected)
        for a, b in zip(before, after):
            if isinstance(a, pd.Series):
                pd.testing.assert_series_equal(a, b)
            else:
                pd.testing.assert_frame_equal(a, b)
        self.assertTrue(after[0].iloc[330:570].all())

    def test_unknown_orientation_is_visible_without_rewriting_tib_validity(self):
        data = self.frame()
        _, episodes, _, _ = correct_belt_inversion(data, walking_detector=lambda *args: pd.DataFrame(columns=["start", "end"]))
        tib = pd.DataFrame({"tib_start": [data.index[0]], "tib_end": [data.index[-1]+pd.Timedelta(milliseconds=10)], "tib_valid": [True]})
        result = annotate_tib_orientation(tib, episodes)
        self.assertTrue(result.orientation_review_required.iloc[0])
        self.assertAlmostEqual(result.orientation_uncertain_wear_hours_in_tib.iloc[0], 120/3600)
        self.assertTrue(result.tib_valid.iloc[0])

    def test_t0_configuration_selects_winner_and_has_no_tilt_option(self):
        from fallspredict_gp_pipeline.config import LowerBackConfig as GP
        from fallspredict_pipeline.config import LowerBackConfig as Ravenna
        for cls in (GP, Ravenna):
            config = cls.from_toml("configs/lower_back_t0_inversion.example.toml")
            self.assertEqual(config.visits, ["T0"])
            self.assertIsNone(config.participant_ids)
            self.assertEqual(config.orientation.method, "inversion")
            self.assertEqual(config.nonwear.window_seconds, 1800.)
            self.assertEqual(config.nonwear.step_seconds, 60.)
            self.assertEqual(config.respiration.min_nonwear_bout_seconds, 900.)
        with self.assertRaisesRegex(ValueError, "tilt alignment"):
            OrientationOptions.from_mapping({"method": "sequence"})


if __name__ == "__main__":
    unittest.main()
