"""Wear, timestamp and rerun regressions against both cohort entry points."""
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

PACKAGES = ('gp_pipeline', 'ravenna_pipeline')


def recording(start='2024-01-01 12:00', count=400):
    index = pd.date_range(start, periods=count, freq='10ms', name='datetime')
    return pd.DataFrame({'acc_x': 0., 'acc_y': 0., 'acc_z': 9.80665,
                         'gyr_x': 0., 'gyr_y': 0., 'gyr_z': 0., 'wear': True,
                         'timestamp': np.arange(count)/100}, index=index)


def walking_result():
    wb = pd.DataFrame({'start': [10], 'end': [100], 'duration_s': [.9],
                       'n_strides': [10], 'stride_duration_s': [.9], 'cadence_spm': [100.],
                       'stride_length_m': [1.], 'walking_speed_mps': [1.]},
                      index=pd.Index([0], name='wb_id'))
    return {'per_wb_params': wb, 'step_counts': pd.Series([10], index=wb.index)}


class GaitProcessingTests(unittest.TestCase):
    def test_real_gait_detector_accepts_stationary_wear_segments(self):
        for pkg in PACKAGES:
            with self.subTest(package=pkg), tempfile.TemporaryDirectory() as tmp:
                entry = importlib.import_module(pkg + '.lower_back.pipeline').run_gait_pipeline
                entry(recording(count=3000), subject='900001', cohort='HA', sensor_height_m=1., save_folder=tmp)
                daily = pd.read_csv(Path(tmp)/'900001_day_aggregation.csv')
                self.assertEqual(daily.processing_status.iloc[0], 'completed')
                self.assertEqual(daily.step_count.iloc[0], 0)
                self.assertEqual(daily.wb_all__count.iloc[0], 0)

    def test_wear_mask_supports_microsecond_index(self):
        index = pd.date_range('2024-01-01', periods=5, freq='10ms').as_unit('us')
        bouts = pd.DataFrame({'start': [index[1]], 'end': [index[3]]})
        for pkg in PACKAGES:
            module = importlib.import_module(pkg + '.lower_back.preprocessing')
            self.assertEqual(module._build_wear_mask_from_nonwear(index, bouts).tolist(), [True, False, False, False, True])

    def test_wear_segments_and_bout_times_on_nonconsecutive_dates(self):
        day = recording()
        day.loc[day.index[150:250], 'wear'] = False
        later = recording('2024-01-03 12:00', 150)
        gapday = pd.concat([recording('2024-01-04 12:00', 150), recording('2024-01-04 12:00:03', 150)])
        frame = pd.concat([day, later, gapday])
        for pkg in PACKAGES:
            with self.subTest(package=pkg), tempfile.TemporaryDirectory() as tmp:
                entry = importlib.import_module(pkg + '.lower_back.pipeline').run_gait_pipeline
                calls = []
                def processor(data, **kwargs):
                    calls.append(data)
                    self.assertTrue(data.wear.all())
                    np.testing.assert_allclose(np.diff(data.timestamp), .01)
                    return walking_result()
                entry(frame, subject='900001', cohort='HA', sensor_height_m=1., save_folder=tmp,
                      process_gait_data_fn=processor)
                self.assertEqual([len(x) for x in calls], [150]*5)
                summary = pd.read_csv(Path(tmp)/'900001_day_aggregation.csv')
                self.assertEqual(summary.step_count.tolist(), [20, 10, 20])
                self.assertEqual(summary.wb_all__count.tolist(), [2, 1, 2])
                self.assertAlmostEqual(summary.nonwear_time_minutes.iloc[0], 1/60)
                dated = pd.read_csv(Path(tmp)/'900001_walking_bouts_datetime.csv', parse_dates=['start', 'end'])
                expected = [day.index[10], day.index[260], later.index[10], gapday.index[10], gapday.index[160]]
                self.assertEqual(dated.start.tolist(), expected)
                self.assertEqual((dated.end-dated.start).dt.total_seconds().tolist(), [.9]*5)
                wb = pd.read_csv(Path(tmp)/'900001_per_wb.csv')
                self.assertEqual(wb.start.tolist(), [10, 260, 10, 10, 160])

    def test_no_gait_is_zero_and_all_nonwear_is_unprocessed(self):
        frame = pd.concat([recording(count=150), recording('2024-01-02', 150)])
        frame.loc[frame.index.date == pd.Timestamp('2024-01-02').date(), 'wear'] = False
        for pkg in PACKAGES:
            with self.subTest(package=pkg), tempfile.TemporaryDirectory() as tmp:
                processor = lambda *args, **kwargs: {'per_wb_params': None}
                entry = importlib.import_module(pkg + '.lower_back.pipeline').run_gait_pipeline
                entry(frame, subject='900001', cohort='HA', sensor_height_m=1., save_folder=tmp,
                      process_gait_data_fn=processor)
                daily = pd.read_csv(Path(tmp)/'900001_day_aggregation.csv')
                self.assertEqual(daily.processing_status.tolist(), ['completed', 'insufficient_continuous_wear'])
                for column in ('step_count', 'total_walking_duration_min', 'wb_all__count'):
                    self.assertEqual(daily[column].iloc[0], 0)
                    self.assertTrue(pd.isna(daily[column].iloc[1]))
                self.assertTrue(pd.isna(daily.wb_all__cadence_spm__avg.iloc[0]))
                self.assertTrue(pd.read_csv(Path(tmp)/'900001_per_wb.csv').empty)
                self.assertTrue(pd.read_csv(Path(tmp)/'900001_walking_bouts_datetime.csv').empty)

    def test_zero_gait_days_contribute_to_amount_average(self):
        for pkg in PACKAGES:
            with self.subTest(package=pkg), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'900001'/'T0'/'McRoberts'/'gait'
                path.mkdir(parents=True)
                pd.DataFrame({'day': [1, 2], 'hours': [20, 20], 'nonwear_time_minutes': [0, 0],
                              'processing_status': ['completed']*2, 'step_count': [100, 0],
                              'wb_all__count': [2, 0], 'wb_all__cadence_spm__avg': [100, np.nan],
                              'unprocessed_wear_seconds': [0, 0]}).to_csv(path/'900001_day_aggregation.csv', index=False)
                aggregate = importlib.import_module(pkg + '.aggregation.gait').aggregate_gait
                summary, _ = aggregate(tmp)
                self.assertEqual(summary.gait_step_count.iloc[0], 50)
                self.assertEqual(summary.gait_wb_all__count.iloc[0], 1)
                self.assertEqual(summary.gait_wb_all__cadence_spm__avg.iloc[0], 100)
                self.assertNotIn('gait_unprocessed_wear_seconds', summary)

    def test_failed_rerun_clears_old_outputs_and_surfaces_error(self):
        for pkg in PACKAGES:
            with self.subTest(package=pkg), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name in ('per_wb', 'day_aggregation', 'walking_bouts_datetime'):
                    (root/f'900001_{name}.csv').write_text('stale')
                entry = importlib.import_module(pkg + '.lower_back.pipeline').run_gait_pipeline
                def fail(*args, **kwargs):
                    raise RuntimeError('synthetic algorithm failure')
                with self.assertRaisesRegex(RuntimeError, 'synthetic algorithm failure'):
                    entry(recording(count=150), subject='900001', cohort='HA', sensor_height_m=1., save_folder=tmp,
                          process_gait_data_fn=fail)
                self.assertFalse((root/'900001_day_aggregation.csv').exists())
                self.assertEqual(pd.read_csv(root/'900001_gait_status.csv').status.iloc[-1], 'failed')

    def test_preprocessing_preserves_elapsed_time_across_gap(self):
        import polars as pl
        from types import SimpleNamespace
        index = pd.to_datetime(['2024-01-01', '2024-01-01 00:00:00.010', '2024-01-01 00:00:02.000'], format='mixed')
        raw = pd.DataFrame({'datetime': index, 'ax': 0., 'ay': 0., 'az': 1., 'gx': 0., 'gy': 0., 'gz': 0.})
        raw.attrs['sample_rate_hz'] = 100
        for pkg in PACKAGES:
            module = importlib.import_module(pkg + '.lower_back.preprocessing')
            nonwear = importlib.import_module(pkg + '.lower_back.nonwear.vanhees2013')
            with self.subTest(package=pkg), patch.object(module, 'read_dp7_as_dataframe_fast', return_value=raw.copy()), \
                 patch.object(module, 'autocalibrate', side_effect=lambda df, **kw: {'df': df}), \
                 patch.object(nonwear, 'vanhees2013', return_value=SimpleNamespace(nonwear_df=pl.DataFrame(), window_df=pl.DataFrame())):
                result = module.preprocess_lowerback_file('synthetic.OMX', verbose=False)
                self.assertEqual(result['df'].timestamp.tolist(), [0., .01, 2.])
                self.assertTrue(result['df'].wear.all())

    def test_height_requires_exact_baseline_and_consistent_values(self):
        for pkg in PACKAGES:
            fn = importlib.import_module(pkg + '.lower_back.gait.gait_functions').get_sensor_height
            df = pd.DataFrame({'patient_id': ['900001.0', '900001', '9000012'],
                               'redcap_event_name': ['baseline_arm_1', 'followup_arm_1', 'baseline_arm_1'],
                               'Height': [180., 170., 160.]})
            self.assertAlmostEqual(fn(df, subject='0900001'), .491*1.8+.2)
            with self.assertRaises(ValueError):
                fn(df, subject='900002')
            with self.assertRaises(ValueError):
                fn(pd.concat([df, df.iloc[:1].assign(Height=150)]), subject='900001')


if __name__ == '__main__':
    unittest.main()
