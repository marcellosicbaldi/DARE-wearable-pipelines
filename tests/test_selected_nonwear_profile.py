"""Selected-profile equivalence and timestamp contracts on invented signals."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl

from scripts import benchmark_lower_back_nonwear as benchmark
from dare_wearables.lower_back.nonwear.ggir2023 import NonwearOptions, detect_nonwear, _epoch_scores
from dare_wearables.lower_back.nonwear.minute_grid import minute_scores
from dare_wearables.lower_back.nonwear.respiration import (
    RespirationOptions, RespirationSettings, analyze_gyro, _merge, _subtract, mask_bounds,
)
from dare_wearables.lower_back.preprocessing import preprocess_lowerback_file


SELECTED = NonwearOptions(window_seconds=1800., step_seconds=60.)


class SelectedProfileTests(unittest.TestCase):
    def test_selected_scores_and_physical_gap_rules_equal_frozen_benchmark(self):
        for fs in (5, 100):
            n = 4*3600*fs+17
            t = np.arange(n)/fs
            axes = [.09*np.sin(.7*t+i)+i*.3 for i in range(3)]
            for i in range(3):
                axes[i][(t>=3600)&(t<6000)] = i*.3
                axes[i][t>=3*3600] = i*.3
            times = pd.date_range('2024-01-01', periods=n, freq=pd.Timedelta(seconds=1/fs))
            frame = pl.DataFrame({'ts': times.to_numpy(), **dict(zip('xyz', axes))})
            for required, post in ((2, True), (2, False), (3, True)):
                options = replace(SELECTED, axes_required=required, postprocessing=post)
                result = detect_nonwear(frame, freq=fs, options=options)
                expected = benchmark.fast_scores(axes, fs, 30, 1)
                np.testing.assert_array_equal(result.window_df['nonwear_score'].to_numpy(), expected)
                np.testing.assert_array_equal(result.window_df['is_final_nonwear'].to_numpy(),
                    benchmark.decisions(expected, required, 1, post))
                self.assertEqual(result.provenance['step_seconds'], 60.)
                self.assertFalse(result.provenance['reference_equivalent'])

    def test_minute_grid_clipping_and_uncommon_rate_equal_direct_reference(self):
        for fs in (5, 31, 100):
            n = 80*60*fs+13
            t = np.arange(n)/fs
            x = .02*np.sin(t)
            x[1] = 12.
            x[60*fs+1] = 8.
            medium = 60*fs
            actual = minute_scores([x]*3, fs, medium, n//medium, window_epochs=30)
            expected = _epoch_scores([x]*3, fs, medium, n//medium, window_epochs=30)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)

    def test_minute_grid_strict_threshold_edges(self):
        n = 30*60*5
        limit = .013*np.sqrt((n-1)/n)
        for value in (np.nextafter(limit, 0), limit, np.nextafter(limit, np.inf)):
            x = np.resize([-value, value], n)
            expected = _epoch_scores([x]*3, 5, 300, 30, window_epochs=30)[0]
            actual = minute_scores([x]*3, 5, 300, 30, window_epochs=30)[0]
            np.testing.assert_array_equal(actual, expected)

    def test_final_selected_mask_matches_benchmark_including_plain_gaps(self):
        fs = 100
        n = 80*60*fs
        t = np.arange(n)/fs
        motion = .1*np.sin(t)
        motion[(t>=15*60)&(t<65*60)] = 0
        times = pd.date_range('2024-01-01', periods=n, freq='10ms').asi8.copy()
        cut = 40*60*fs
        times[cut:] += 2_000_000_000
        raw = pd.DataFrame({'datetime': pd.to_datetime(times), 'ax': motion, 'ay': motion, 'az': motion+1,
                            'gx': 0., 'gy': 0., 'gz': 0.})
        raw.loc[:10*60*fs, 'gy'] = .06*np.sin(2*np.pi*19/60*t[:10*60*fs+1])
        options = RespirationOptions(enabled=True, window_seconds=30., min_nonwear_bout_seconds=900.)
        with patch('dare_wearables.lower_back.preprocessing.read_dp7_as_dataframe_fast', return_value=raw.copy()), \
             patch('dare_wearables.lower_back.preprocessing.autocalibrate', side_effect=lambda df, **kw: {'df':df}):
            prepared = preprocess_lowerback_file('invented.omx', verbose=False, nonwear=SELECTED,
                                                 respiration=options, protocol_hours=168.)
        segments = np.array([[0, cut], [cut, n]])
        self.assertEqual(prepared['df'].attrs['processing_segments'], segments.tolist())
        base = []
        for a, b in segments:
            axes = [raw[name].to_numpy()[a:b] for name in ('ax','ay','az')]
            scores = benchmark.fast_scores(axes, fs, 30, 1)
            base.append(benchmark.mask_bouts(benchmark.decisions(scores, 2, 1), 60*fs, a))
        result = analyze_gyro(raw, fs, window_seconds=30., overlap=.5, settings=RespirationSettings(),
            axis_columns=('gx','gy','gz'), time_column='datetime', segment_bounds=segments, store_psd=False)
        evidence = result.windows.loc[result.windows.respiration_present,['start_sample','stop_sample']].to_numpy()
        expected = benchmark.split_bounds(_subtract(np.concatenate(base), evidence), segments)
        expected = expected[(expected[:,1]-expected[:,0])/fs >= 900]
        np.testing.assert_array_equal(mask_bounds(~prepared['df'].wear.to_numpy()), _merge(expected))
        self.assertEqual(prepared['respiration']['result'].window_seconds, 30.)

    def test_explicit_elapsed_cap_preserves_gaps_and_legacy_default(self):
        raw = pd.DataFrame({'datetime':pd.to_datetime(['2024-01-01 00:00:00','2024-01-01 00:00:00.010',
            '2024-01-01 00:59:59','2024-01-01 01:00:00'],format='mixed'),
            'ax':0.,'ay':0.,'az':1.,'gx':0.,'gy':0.,'gz':0.})
        with patch('dare_wearables.lower_back.preprocessing.read_dp7_as_dataframe_fast', return_value=raw), \
             patch('dare_wearables.lower_back.preprocessing.autocalibrate', side_effect=lambda df, **kw:{'df':df}):
            new = preprocess_lowerback_file('invented.omx',verbose=False,nonwear=SELECTED,max_samples=2,protocol_hours=1)
            old = preprocess_lowerback_file('invented.omx',verbose=False,max_samples=2)
        self.assertEqual(len(new['df']),3)
        self.assertEqual(len(old['df']),2)
        self.assertEqual(new['df'].timestamp.iloc[-1],3599.)

    def test_both_cohort_configurations_accept_selected_profile_and_preserve_defaults(self):
        from fallspredict_gp_pipeline.config import LowerBackConfig as GP
        from fallspredict_pipeline.config import LowerBackConfig as Ravenna
        source = Path('configs/lower_back_selected_full.example.toml').read_text()
        for cls in (GP,Ravenna):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'config.toml';path.write_text(source)
                config=cls.from_toml(path)
                self.assertEqual(config.nonwear,SELECTED)
                self.assertEqual(config.protocol_hours,168.)
                self.assertEqual(config.respiration.window_seconds,30.)
        self.assertEqual(NonwearOptions().step_seconds,900.)
        self.assertEqual(NonwearOptions().window_seconds,3600.)
        for bad in ({'step_seconds':30},{'axes_required':1},{'postprocessing':'false'}):
            with self.assertRaises(ValueError):NonwearOptions.from_mapping(bad)

    def test_standard_gp_example_selects_reviewed_t0_profile_without_changing_ravenna(self):
        from fallspredict_gp_pipeline.config import LowerBackConfig as GP
        from fallspredict_pipeline.config import LowerBackConfig as Ravenna
        gp = GP.from_toml('configs/lower_back.example.toml')
        reviewed = GP.from_toml('configs/lower_back_t0_inversion.example.toml')
        ravenna = Ravenna.from_toml('configs/ravenna_lower_back.example.toml')
        self.assertEqual(gp.visits, ['T0'])
        self.assertEqual(gp.protocol_hours, 168.)
        self.assertEqual(gp.nonwear, SELECTED)
        self.assertEqual(gp.nonwear, reviewed.nonwear)
        self.assertEqual(gp.respiration, reviewed.respiration)
        self.assertEqual(gp.orientation, reviewed.orientation)
        self.assertEqual(gp.timestamps, reviewed.timestamps)
        self.assertTrue(gp.respiration.enabled)
        self.assertEqual(gp.respiration.min_nonwear_bout_seconds, 900.)
        self.assertEqual(gp.orientation.method, 'inversion')
        self.assertFalse(ravenna.respiration.enabled)
        self.assertEqual(ravenna.nonwear, NonwearOptions())


if __name__=='__main__':
    unittest.main()
