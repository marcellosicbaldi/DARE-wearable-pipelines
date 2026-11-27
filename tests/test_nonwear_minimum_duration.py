"""Final-duration boundaries, missing-time safety, and production mask audits."""
from dataclasses import replace
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dare_wearables.lower_back.nonwear.audit import daily_summary, output_record, write_audit, audit_signature
from dare_wearables.lower_back.nonwear.respiration import (
    RespirationOptions, analyze_gyro, apply_override, filter_short_nonwear_bouts,
)
from dare_wearables.lower_back.preprocessing import _build_wear_mask_from_nonwear, preprocess_lowerback_file
from scripts.refine_lower_back_nonwear import classify_episodes, refine_assessments


class MinimumDurationTests(unittest.TestCase):
    def test_strict_cutoff_exactly_15_minutes_is_retained(self):
        bouts = pd.DataFrame({"duration_seconds": [0.01, 899.99, 900., 900.01]})
        kept, short = filter_short_nonwear_bouts(bouts, 900.)
        self.assertEqual(kept.duration_seconds.tolist(), [900., 900.01])
        self.assertEqual(short.duration_seconds.tolist(), [.01, 899.99])
        pd.testing.assert_frame_equal(bouts, filter_short_nonwear_bouts(bouts, 0)[0])
        empty = pd.read_csv(StringIO('duration_seconds\n'))
        self.assertTrue(filter_short_nonwear_bouts(empty, 900)[0].empty)
        for value in (-1, np.inf, np.nan, True, "900"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RespirationOptions.from_mapping({"min_nonwear_bout_seconds": value})

    def test_filter_runs_after_gap_splitting_without_joining_segments(self):
        fs = 10
        times = pd.date_range("2024-01-01", periods=1200*fs, freq="100ms").to_numpy()
        times[600*fs:] += np.timedelta64(60, "s")
        gyro = pd.DataFrame({"gx": 0., "gy": 0., "gz": 0.}, index=pd.DatetimeIndex(times))
        analysis = analyze_gyro(gyro, fs, store_psd=False)
        baseline = pd.DataFrame({"start_sample": [0], "stop_sample": [len(gyro)]})
        changed = apply_override(analysis, baseline, gyro.index, min_nonwear_bout_seconds=900)
        self.assertEqual(changed["nonwear_bouts_before_duration_filter"].duration_seconds.tolist(), [600., 600.])
        self.assertEqual(changed["short_nonwear_bouts"].duration_seconds.tolist(), [600., 600.])
        self.assertTrue(changed["nonwear_bouts"].empty)
        self.assertAlmostEqual(changed["summary"]["short_nonwear_reassigned_to_wear_hours"], 1200/3600)
        self.assertEqual((changed["short_nonwear_bouts"].start.iloc[1] -
                          changed["short_nonwear_bouts"].end.iloc[0]).total_seconds(), 60.)

    def test_production_mask_and_audit_separate_the_two_reassignment_reasons(self):
        fs = 100
        n = 1800*fs
        raw = pd.DataFrame({"datetime": pd.date_range("2024-01-01 08:45", periods=n, freq="10ms"),
                            "ax": 0., "ay": 0., "az": 1., "gx": 0., "gy": 0., "gz": 0.})
        analysis = analyze_gyro(raw, fs, axis_columns=("gx", "gy", "gz"), time_column="datetime", store_psd=False)
        analysis.windows.loc[29, "respiration_present"] = True  # [870, 930): two residual 870 s bouts.
        options = RespirationOptions(enabled=True, min_nonwear_bout_seconds=900)
        with patch("dare_wearables.lower_back.preprocessing.read_dp7_as_dataframe_fast", return_value=raw), \
             patch("dare_wearables.lower_back.preprocessing.autocalibrate", side_effect=lambda df, **kw: {"df": df}), \
             patch("dare_wearables.lower_back.nonwear.respiration.analyze_gyro", return_value=analysis):
            prepared = preprocess_lowerback_file("synthetic", verbose=False, respiration=options)
        frame, diag = prepared["df"], prepared["respiration"]
        self.assertFalse(frame.wear_accelerometer.any())
        self.assertTrue(frame.wear.all())
        self.assertEqual(frame.wear_respiration_override.sum(), 60*fs)
        self.assertEqual(frame.wear_short_nonwear_override.sum(), 1740*fs)
        self.assertFalse((frame.wear_respiration_override & frame.wear_short_nonwear_override).any())
        self.assertTrue(prepared["nonwear_bouts"].empty)
        np.testing.assert_array_equal(frame.wear, _build_wear_mask_from_nonwear(frame.datetime, prepared["nonwear_bouts"]))
        daily = daily_summary(frame, fs)
        self.assertAlmostEqual(daily.reassigned_to_wear_hours.sum(), .5)
        self.assertAlmostEqual(daily.respiration_reassigned_to_wear_hours.sum(), 60/3600)
        self.assertAlmostEqual(daily.short_nonwear_reassigned_to_wear_hours.sum(), 1740/3600)
        self.assertAlmostEqual(daily.reassigned_09_21_hours.sum(), .25)
        summary = diag["summary"]
        self.assertEqual(summary["revised_nonwear_hours"], 0)
        self.assertEqual(summary["reassigned_to_wear_hours"], .5)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'synthetic.omx'
            source.write_bytes(b'synthetic')
            summary = write_audit(prepared, root/'audit', source=source, options=options)
            self.assertEqual(summary['short_nonwear_bouts_reassigned'], 2)
            saved = pd.read_csv(root/'audit/short_nonwear_reassigned_intervals.csv')
            self.assertEqual(saved.duration_seconds.tolist(), [870., 870.])
            self.assertTrue(pd.read_csv(root/'audit/final_nonwear_intervals.csv').empty)
            key = audit_signature(source, options)[0]
            self.assertTrue(output_record(root/'audit', options).reusable(key))
            self.assertNotEqual(key, audit_signature(source, replace(options, min_nonwear_bout_seconds=0))[0])

    def test_both_cohort_configs_load_the_final_rule(self):
        from fallspredict_gp_pipeline.config import LowerBackConfig as GP
        from fallspredict_pipeline.config import LowerBackConfig as Ravenna
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'config.toml'
            path.write_text(f'[lower_back]\nbronze_root="{temporary}"\nsilver_root="{temporary}/out"\n'
                            '[lower_back.respiration]\nenabled=true\nmin_nonwear_bout_seconds=900\n')
            for cls in (GP, Ravenna):
                self.assertEqual(cls.from_toml(path).respiration.min_nonwear_bout_seconds, 900)

    def test_day_night_assignment_handles_boundaries_and_multiple_days(self):
        frame = pd.DataFrame({'start': pd.to_datetime(['2024-01-01 08:00', '2024-01-01 22:00',
                                                      '2024-01-01 07:55', '2024-01-01 00:00']),
                              'end': pd.to_datetime(['2024-01-01 22:00', '2024-01-02 08:00',
                                                    '2024-01-01 08:15', '2024-01-03 00:00'])})
        result = classify_episodes(frame)
        self.assertEqual(result.period.tolist(), ['day', 'night', 'day', 'day'])
        self.assertEqual(result.night_wall_seconds.tolist(), [0, 10*3600, 300, 20*3600])
        self.assertEqual(result.crosses_day_night_boundary.tolist(), [False, False, True, True])

    def test_saved_results_deduplicate_controls_and_reject_changed_outputs(self):
        from dare_wearables.common.output_state import _digest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root/'source'
            folder = run/'0001/T0/McRoberts/nonwear'
            folder.mkdir(parents=True)
            # One exact-cutoff night bout and one 899-second daytime bout.
            bouts = pd.DataFrame({'start_sample': [0, 100000], 'stop_sample': [90000, 189900],
                                  'start': ['2024-01-01 00:00:00', '2024-01-01 12:00:00'],
                                  'end': ['2024-01-01 00:15:00', '2024-01-01 12:14:59'],
                                  'duration_seconds': [900., 899.]})
            bouts.to_csv(folder/'final_nonwear_intervals.csv', index=False)
            summary = {'baseline_method': 'ggir_2023_matched', 'sample_rate_hz': 100,
                       'n_samples': 200000, 'revised_nonwear_hours': 1799/3600,
                       'baseline_nonwear_hours': 1., 'observed_hours': 200000/100/3600,
                       'reassigned_to_wear_hours': 1.-1799/3600}
            (folder/'nonwear_summary.json').write_text(json.dumps(summary))
            hashes = {name: _digest(folder/name) for name in ['final_nonwear_intervals.csv','nonwear_summary.json']}
            (folder/'respiration_manifest.json').write_text(json.dumps({'outputs': hashes, 'signature': 'synthetic'}))
            item = {'subject': '0001', 'visit': 'T0', 'sensor': 'McRoberts', 'status': 'completed'}
            (run/'assessment.json').write_text(json.dumps({'recordings':[item]}))
            before_hash = _digest(folder/'final_nonwear_intervals.csv')
            with redirect_stdout(StringIO()):
                result = refine_assessments([run, run], root/'out', expected_recordings=1)
            statistics = json.loads((result/'statistics.json').read_text())
            self.assertEqual(statistics['stages']['after_filter']['episodes'], 1)
            self.assertEqual(statistics['stages']['after_filter']['night_episodes'], 1)
            self.assertEqual(statistics['stages']['after_filter']['nonwear_hours_total'], .25)
            self.assertEqual(json.loads((result/'duration_filter_manifest.json').read_text())['duplicate_controls_not_counted'], 1)
            self.assertEqual(_digest(folder/'final_nonwear_intervals.csv'), before_hash)
            (folder/'final_nonwear_intervals.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                refine_assessments([run], root/'out')


if __name__ == '__main__':
    unittest.main()
