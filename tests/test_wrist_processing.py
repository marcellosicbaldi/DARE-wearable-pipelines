"""Synthetic wrist pipeline regressions, with models replaced by deterministic outputs."""
from contextlib import ExitStack
import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import polars as pl

PACKAGES = ('gp_pipeline', 'ravenna_pipeline')


def sensor_frames(frequency=64, seconds=120):
    index = pd.date_range('2024-01-01 22:00', periods=frequency*seconds+1,
                          freq=pd.Timedelta(seconds=1/frequency), tz='Europe/Rome')
    index = index.as_unit('us')
    return pd.DataFrame({'x': 0., 'y': 0., 'z': 1.}, index=index), pd.DataFrame({'ppg': 0.}, index=index)


def sleep_window(start, end):
    return pd.DataFrame({'night_id': [1], 'source_night_id': [1], 'sleep_window_source': ['sleep_diary'],
                         'priority_rank': [0], 'spt_start': [start], 'spt_end': [end]})


class WristProcessingTests(unittest.TestCase):
    def test_real_burst_detection_preserves_timezone_and_handles_flat_signal(self):
        acc, _ = sensor_frames()
        values = np.ones(len(acc))
        values[20*64:40*64] += .2*np.sin(np.arange(20*64)/64*2*np.pi*2)
        for pkg in PACKAGES:
            module = importlib.import_module(pkg+'.wrist.heart_rate_variability.detect_acc_bursts')
            flat = module.detect_bursts(pd.Series(1., index=acc.index), sampling_rate=64, alfa=.035)
            self.assertTrue(flat.empty)
            bursts = module.detect_bursts(pd.Series(values, index=acc.index), sampling_rate=64, alfa=.035)
            self.assertFalse(bursts.empty)
            self.assertEqual(str(bursts.start.dt.tz), 'Europe/Rome')
            self.assertEqual(str(bursts.end.dt.tz), 'Europe/Rome')
            self.assertTrue((bursts.end > bursts.start).all())

    def test_default_output_locations_match_aggregation_inputs(self):
        for pkg in PACKAGES:
            hr = importlib.import_module(pkg+'.wrist.heart_rate.config').HeartRateConfig
            hrv = importlib.import_module(pkg+'.wrist.heart_rate_variability.config').HRVConfig
            self.assertEqual(hr(input_root=Path('synthetic'), output_root=Path('synthetic')).output_subdir, 'beliefppg')
            self.assertEqual(hrv(input_root=Path('synthetic'), output_root=Path('synthetic')).output_subdir, 'hrv')

    def test_timezone_conversion_preserves_instants_and_local_clock_times(self):
        for pkg in PACKAGES:
            module = importlib.import_module(pkg+'.common.recording')
            acc, _ = sensor_frames()
            utc = acc.tz_convert('UTC')
            converted = module.in_timezone(utc, 'Europe/Rome')
            self.assertEqual(converted.index[0], acc.index[0])
            naive = acc.tz_localize(None)
            self.assertEqual(module.in_timezone(naive, 'Europe/Rome').index[0], acc.index[0])
            self.assertEqual(module.timestamp_in_timezone(utc.index[0], 'Europe/Rome'), acc.index[0])

    def test_sleep_labels_accept_different_timestamp_units(self):
        for pkg in PACKAGES:
            for filename, function, label in (('vh2015_sib', 'label_epochs_from_bouts', 'sib_sleep'),
                                               ('vh2018_spt', 'label_epochs_from_windows', 'in_spt')):
                with self.subTest(package=pkg, function=function):
                    module = importlib.import_module(pkg+'.wrist.sleep.'+filename)
                    epochs = pl.DataFrame({'ts': pd.date_range('2024-01-01', periods=3, freq='1s')}).with_columns(pl.col('ts').cast(pl.Datetime('us')))
                    bouts = pl.DataFrame({'start': [pd.Timestamp('2024-01-01 00:00:01')],
                                          'end': [pd.Timestamp('2024-01-01 00:00:02')]}).with_columns(pl.all().cast(pl.Datetime('ns')))
                    actual = getattr(module, function)(epochs, bouts, start_col='start', end_col='end', label_col=label)
                    self.assertEqual(actual[label].to_list(), [False, True, True])

    def test_combined_skipped_stages_remove_previous_results(self):
        cases = [('gp_pipeline', 'gp', 'preprocess_empatica_recording',
                  {'acc_parquet_path': 'synthetic.parquet', 'temp_parquet_path': 'synthetic_temp.parquet'}),
                 ('ravenna_pipeline', 'ravenna', 'preprocess_geneactiv_recording',
                  {'geneactiv_bin_path': 'synthetic.bin'})]
        for pkg, cohort, prep, inputs in cases:
            module = importlib.import_module(pkg+'.wrist.sleep.sleep_pipeline_'+cohort)
            with self.subTest(package=pkg), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                for name in ('circadian_metrics.csv', 'activity_intensity_summary.csv', 'sleep_output_old.csv'):
                    (root/name).write_text('old')
                (root/'unrelated.txt').write_text('keep')
                processing = importlib.import_module('dare_wearables.wrist.sleep.pipeline')
                preprocessed = {k: pd.DataFrame() for k in ('calibrated_df', 'acc_df', 'temp_df', 'nonwear_df')}
                preprocessed['info'] = {}
                stack.enter_context(patch.object(module, prep, return_value=preprocessed))
                stack.enter_context(patch.object(processing, 'run_sleep_pipeline_gp_from_preprocessed',
                                                 return_value={'sleep_outputs_by_source': {}, 'info': {}}))
                stack.enter_context(patch.object(processing, 'run_circadian_pipeline_gp_from_preprocessed',
                                                 side_effect=ValueError('Only 1 valid days found, but min_valid_days=3.')))
                stack.enter_context(patch.object(processing, '_select_activity_sleep_windows', return_value=pd.DataFrame()))
                result = getattr(module, 'run_sleep_and_circadian_pipeline_'+cohort)(**inputs, participant='900001', save_folder=root)
                self.assertEqual([x['status'] for x in result['stage_status']], ['completed', 'skipped', 'skipped'])
                self.assertFalse((root/'circadian_metrics.csv').exists())
                self.assertFalse((root/'activity_intensity_summary.csv').exists())
                self.assertFalse((root/'sleep_output_old.csv').exists())
                self.assertEqual((root/'unrelated.txt').read_text(), 'keep')

    def test_combined_exception_cleans_partial_current_outputs(self):
        for pkg in PACKAGES:
            module = importlib.import_module(pkg+'.wrist.sleep.sleep_pipeline_gp')
            with self.subTest(package=pkg), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                def fail(*args, **kwargs):
                    (root/'sleep_output.csv').write_text('partial')
                    raise RuntimeError('synthetic failure')
                processing = importlib.import_module('dare_wearables.wrist.sleep.pipeline')
                with patch.object(processing, 'preprocess_empatica_recording', side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                        module.run_sleep_and_circadian_pipeline_gp('synthetic.parquet', 'temp.parquet', participant='900001', save_folder=root)
                self.assertFalse((root/'sleep_output.csv').exists())

    def test_recording_intersections_use_individual_sampling_rates(self):
        for pkg in PACKAGES:
            module = importlib.import_module(pkg+'.wrist.heart_rate.pipeline')
            acc, _ = sensor_frames(frequency=32, seconds=4)
            _, ppg = sensor_frames(frequency=64, seconds=4)
            # A single missing 64-Hz PPG sample must split, even with 32-Hz ACC.
            ppg = ppg.drop(ppg.index[128])
            aa, pp = module._split_recording_portions(acc, ppg, min_good_portion_minutes=0,
                                                     acc_frequency=32, ppg_frequency=64)
            self.assertEqual(len(aa), 2)
            for a, p in zip(aa, pp):
                np.testing.assert_allclose(np.diff(p.index.to_numpy(dtype="datetime64[ns]").astype(np.int64))/1e9, 1/64)
            with self.assertRaisesRegex(ValueError, 'frequency'):
                module._split_recording_portions(acc, ppg, min_good_portion_minutes=0, acc_frequency=64)

    def test_hrv_flags_exact_interpolated_fraction_and_never_crosses_gaps(self):
        for pkg in PACKAGES:
            with self.subTest(package=pkg), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                module = importlib.import_module(pkg+'.wrist.heart_rate_variability.pipeline')
                root = Path(tmp)
                acc, ppg = sensor_frames(seconds=240)
                ppg = ppg.loc[(ppg.index < acc.index[64*100]) | (ppg.index >= acc.index[64*140])]
                for name in ('ppg', 'acc'):
                    (root/(name+'.parquet')).write_text('synthetic')
                files = {name: root/(name+'.parquet') for name in ('ppg', 'acc')}
                windows = sleep_window(acc.index[0], acc.index[-1])
                stack.enter_context(patch.object(module, 'find_empatica_heart_rate_files', return_value=files))
                stack.enter_context(patch.object(module, 'load_ppg', return_value=ppg))
                stack.enter_context(patch.object(module, 'load_acc', return_value=acc))
                stack.enter_context(patch.object(module, '_load_selected_sleep_windows', return_value=(windows, None, 'synthetic')))
                stack.enter_context(patch.object(module, 'detect_bursts', return_value=pd.DataFrame(columns=['start', 'end'])))
                detector = stack.enter_context(patch.object(module, 'MSPTDfast', side_effect=lambda values, sampling_rate: (None, np.arange(0, len(values), sampling_rate))))
                artifact_info = {'ectopic': [5], 'missed': [], 'extra': [], 'longshort': []}
                stack.enter_context(patch.object(module, 'signal_fixpeaks', return_value=(artifact_info, None)))
                kwargs = dict(input_root=root, output_root=root, participant='900001')
                result = module.run_hrv_pipeline(**kwargs)
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(detector.call_count, 2)
                table = pd.read_csv(result['hrv_output_path'])
                self.assertEqual(len(table), 2)
                self.assertEqual(table.n_interpolated_beats.tolist(), [1, 1])
                self.assertTrue(table.contains_interpolated_beats.all())
                np.testing.assert_allclose(table.interpolated_fraction, 1/table.n_beats)
                self.assertTrue((table.mean_hr == 60).all())
                ibi = pd.read_parquet(result['ibi_output_path'])
                self.assertEqual(ibi.interpolated.sum(), 2)
                self.assertIsNotNone(ibi.index.tz)
                self.assertTrue(ibi.index.is_unique)
                # A complete, unchanged run is reusable.
                module.run_hrv_pipeline(**kwargs)
                self.assertEqual(detector.call_count, 2)
                # A missing main CSV must invalidate the cache even if IBI and bursts remain.
                result['hrv_output_path'].unlink()
                module.run_hrv_pipeline(**kwargs)
                self.assertEqual(detector.call_count, 4)
                # Changed parameters must invalidate the completion record.
                module.run_hrv_pipeline(**kwargs, threshold_bursts=.04)
                self.assertEqual(detector.call_count, 6)

    def test_hrv_empty_and_failed_reruns_do_not_reuse_old_tables(self):
        for pkg in PACKAGES:
            with self.subTest(package=pkg), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                module = importlib.import_module(pkg+'.wrist.heart_rate_variability.pipeline')
                root = Path(tmp)
                acc, ppg = sensor_frames()
                files = {name: root/(name+'.parquet') for name in ('ppg', 'acc')}
                for path in files.values():
                    path.write_text('synthetic')
                stack.enter_context(patch.object(module, 'find_empatica_heart_rate_files', return_value=files))
                load = stack.enter_context(patch.object(module, 'load_ppg', return_value=ppg))
                stack.enter_context(patch.object(module, 'load_acc', return_value=acc))
                select = stack.enter_context(patch.object(module, '_load_selected_sleep_windows', return_value=(sleep_window(acc.index[0], acc.index[-1]), None, 'synthetic')))
                stack.enter_context(patch.object(module, 'detect_bursts', return_value=pd.DataFrame(columns=['start', 'end'])))
                detector = stack.enter_context(patch.object(module, 'MSPTDfast', return_value=(None, [])))
                kwargs = dict(input_root=root, output_root=root, participant='900001')
                result = module.run_hrv_pipeline(**kwargs)
                self.assertEqual(result['status'], 'skipped')
                self.assertTrue(pd.read_csv(result['hrv_output_path']).empty)
                module.run_hrv_pipeline(**kwargs)
                self.assertEqual(detector.call_count, 2)
                result['hrv_output_path'].write_text('stale')
                select.return_value = (pd.DataFrame(), None, 'synthetic')
                module.run_hrv_pipeline(**kwargs)
                self.assertFalse(result['hrv_output_path'].exists())
                result['hrv_output_path'].write_text('stale')
                select.return_value = (sleep_window(acc.index[0], acc.index[-1]), None, 'synthetic')
                load.side_effect = RuntimeError('unreadable input')
                with self.assertRaisesRegex(RuntimeError, 'unreadable input'):
                    module.run_hrv_pipeline(**kwargs)
                self.assertFalse(result['hrv_output_path'].exists())

    def test_heart_rate_correct_thresholds_and_cache_invalidation(self):
        for pkg in PACKAGES:
            with self.subTest(package=pkg), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                module = importlib.import_module(pkg+'.wrist.heart_rate.pipeline')
                root = Path(tmp)
                acc, _ = sensor_frames(frequency=64, seconds=12)
                _, ppg = sensor_frames(frequency=32, seconds=12)
                files = {name: root/(name+'.parquet') for name in ('ppg', 'acc')}
                for path in files.values():
                    path.write_text('synthetic')
                stack.enter_context(patch.object(module, 'find_empatica_ppg_acc_files', return_value=files))
                stack.enter_context(patch.object(module, 'load_ppg', return_value=ppg))
                stack.enter_context(patch.object(module, 'load_acc', return_value=acc))
                inference = Mock(return_value=(np.array([60.]), np.array([1.]), np.array([[2., 4.]])))
                stack.enter_context(patch.dict('sys.modules', {'beliefppg': SimpleNamespace(infer_hr_uncertainty=inference)}))
                kwargs = dict(input_root=root, output_root=root, participant='900001', acc_frequency=64, ppg_frequency=32,
                              min_good_portion_minutes=0, min_inference_minutes=.1)
                result = module.run_heart_rate_pipeline(**kwargs)
                self.assertEqual(result['status'], 'completed')
                table = pd.read_csv(result['output_path'], index_col=0)
                self.assertEqual(pd.Timestamp(table.index[0]), acc.index[0]+pd.Timedelta(seconds=3))
                module.run_heart_rate_pipeline(**kwargs)
                self.assertEqual(inference.call_count, 1)
                result['output_path'].write_text('altered output')
                module.run_heart_rate_pipeline(**kwargs)
                self.assertEqual(inference.call_count, 2)
                files['ppg'].write_text('changed input')
                module.run_heart_rate_pipeline(**kwargs)
                self.assertEqual(inference.call_count, 3)
                inference.return_value = (np.array([]), np.array([]), np.empty((0, 2)))
                empty = module.run_heart_rate_pipeline(**kwargs, skip_existing=False)
                self.assertEqual(empty['status'], 'skipped')
                self.assertTrue(pd.read_csv(empty['output_path']).empty)
                module.run_heart_rate_pipeline(**kwargs)
                self.assertEqual(inference.call_count, 5)
                inference.side_effect = RuntimeError('model failed')
                with self.assertRaisesRegex(RuntimeError, 'model failed'):
                    module.run_heart_rate_pipeline(**kwargs)
                self.assertFalse(result['output_path'].exists())


if __name__ == '__main__':
    unittest.main()
