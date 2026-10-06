"""Architecture and compatibility regressions for the extracted shared core."""
import ast
from contextlib import ExitStack
import importlib
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'src'


class SharedCoreTests(unittest.TestCase):
    def test_core_never_imports_a_cohort_adapter(self):
        for path in (SRC/'dare_wearables').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    names = [node.module or '']
                for name in names:
                    with self.subTest(file=str(path.relative_to(SRC)), dependency=name):
                        self.assertNotIn(name.split('.')[0], ('fallspredict_gp_pipeline', 'fallspredict_pipeline'))

    def test_legacy_algorithm_modules_are_the_same_shared_module(self):
        mappings = {
            'common.autocalibrate': 'common.autocalibrate',
            'common.recording': 'common.recording',
            'common.output_state': 'common.output_state',
            'lower_back.preprocessing': 'lower_back.preprocessing',
            'lower_back.io.mcroberts_loader': 'lower_back.io.mcroberts_loader',
            'lower_back.gait.gait_functions': 'lower_back.gait.functions',
            'lower_back.gait.daily': 'lower_back.gait.daily',
            'lower_back.posture.lying_functions': 'lower_back.posture.lying_functions',
            'wrist.circadian.circadian_pipeline': 'wrist.circadian.metrics',
            'wrist.circadian.circadian_pipeline_gp': 'wrist.circadian.empatica',
            'wrist.circadian.activity_intensity': 'wrist.circadian.activity_intensity',
            'wrist.sleep.sleep_pipeline_gp': 'wrist.sleep.pipeline',
            'wrist.sleep.vh2015_sib': 'wrist.sleep.vh2015_sib',
            'wrist.sleep.vh2018_spt': 'wrist.sleep.vh2018_spt',
            'wrist.data_io.geneactiv': 'wrist.data_io.geneactiv',
            'wrist.heart_rate.pipeline': 'wrist.heart_rate.pipeline',
            'wrist.heart_rate_variability.pipeline': 'wrist.heart_rate_variability.pipeline',
        }
        for old, new in mappings.items():
            shared = importlib.import_module('dare_wearables.'+new)
            for cohort in ('fallspredict_gp_pipeline', 'fallspredict_pipeline'):
                with self.subTest(cohort=cohort, module=old):
                    legacy = importlib.import_module(cohort+'.'+old)
                    if cohort == 'fallspredict_pipeline' and old == 'wrist.sleep.sleep_pipeline_gp':
                        self.assertIs(legacy.run_wrist_from_preprocessed, shared.run_wrist_from_preprocessed)
                    else:
                        self.assertIs(legacy, shared)
        old = importlib.import_module('fallspredict_pipeline.wrist.circadian.geneactiv_preprocessing')
        self.assertIs(old, importlib.import_module('dare_wearables.wrist.circadian.geneactiv_preprocessing'))

    def test_ravenna_legacy_empatica_positional_arguments_keep_their_meaning(self):
        legacy = importlib.import_module('fallspredict_pipeline.wrist.sleep.sleep_pipeline_gp')
        shared = importlib.import_module('dare_wearables.wrist.sleep.pipeline')
        with patch.object(shared, 'run_sleep_pipeline_gp', return_value={'sleep': True}) as runner:
            result = legacy.run_sleep_pipeline_gp('acc.parquet', 'temp.parquet', '900001', 'T1', 'pyarrow', 'vanhees2013')
        self.assertEqual(result, {'sleep': True})
        self.assertEqual(runner.call_args.kwargs['parquet_engine'], 'pyarrow')
        self.assertEqual(runner.call_args.kwargs['nonwear_method'], 'vanhees2013')
        self.assertEqual(runner.call_args.kwargs['visit'], 'T1')
        self.assertNotIn('recruitment_tracker_path', runner.call_args.kwargs)

    def test_core_imports_without_either_cohort_package(self):
        script = '''
import importlib.abc, importlib, sys
class NoCohorts(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('fallspredict_gp_pipeline', 'fallspredict_pipeline'):
            raise ImportError('Unexpected cohort dependency: ' + fullname)
sys.meta_path.insert(0, NoCohorts())
for name in (
    'dare_wearables.common.paths',
    'dare_wearables.lower_back.pipeline',
    'dare_wearables.wrist.sleep.pipeline',
    'dare_wearables.wrist.circadian.geneactiv_preprocessing',
    'dare_wearables.wrist.heart_rate.pipeline',
    'dare_wearables.wrist.heart_rate_variability.pipeline',
    'dare_wearables.aggregation.cohorts',
):
    importlib.import_module(name)
assert not any(name.split('.')[0] in ('fallspredict_gp_pipeline', 'fallspredict_pipeline') for name in sys.modules)
'''
        with tempfile.TemporaryDirectory() as tmp:
            import os
            env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE='1')
            completed = subprocess.run([sys.executable, '-c', script], cwd=tmp, env=env,
                                       capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_each_sensor_adapter_calls_the_same_wrist_stage_sequence(self):
        shared = importlib.import_module('dare_wearables.wrist.sleep.pipeline')
        ravenna = importlib.import_module('fallspredict_pipeline.wrist.sleep.sleep_pipeline_ravenna')
        prepared = {'calibrated_df': object(), 'acc_df': object(), 'temp_df': object(),
                    'nonwear_df': object(), 'info': {}}
        for module, function, loader, inputs, method in (
            (shared, 'run_wrist_from_empatica', 'preprocess_empatica_recording',
             {'acc_parquet_path': Path('acc.parquet'), 'temp_parquet_path': Path('temp.parquet')}, 'empatica_detach'),
            (ravenna, 'run_sleep_and_circadian_pipeline_ravenna', 'preprocess_geneactiv_recording',
             {'geneactiv_bin_path': Path('sensor.bin')}, 'vanhees2013'),
        ):
            with self.subTest(function=function), ExitStack() as stack:
                prep = stack.enter_context(patch.object(module, loader, return_value=prepared))
                process = stack.enter_context(patch.object(shared, 'run_wrist_from_preprocessed', return_value={'shared': True}))
                # The adapter imports the stage function directly; patch that
                # reference too when the adapter has its own module globals.
                if module is ravenna:
                    stack.enter_context(patch.object(ravenna, 'run_wrist_from_preprocessed', process))
                result = getattr(module, function)(**inputs, participant='900001', min_valid_days=4, activity_epoch_seconds=10)
                self.assertEqual(result, {'shared': True})
                prep.assert_called_once()
                process.assert_called_once()
                self.assertIs(process.call_args.kwargs['preprocessed'], prepared)
                self.assertEqual(process.call_args.kwargs['nonwear_method'], method)
                self.assertEqual(process.call_args.kwargs['min_valid_days'], 4)
                self.assertEqual(process.call_args.kwargs['activity_epoch_seconds'], 10)

    def test_cohort_defaults_and_legacy_gait_policy_are_preserved(self):
        bo = importlib.import_module('fallspredict_gp_pipeline.aggregation.sleep')
        ra = importlib.import_module('fallspredict_pipeline.aggregation.sleep')
        self.assertEqual(bo.DEFAULT_SLEEP_SUBDIR, Path('Empatica/sleep_circadian'))
        self.assertEqual(ra.DEFAULT_SLEEP_SUBDIR, Path('GENEActiv/sleep_circadian'))
        for cohort, legacy, threshold in (('fallspredict_gp_pipeline', True, 1024*1024), ('fallspredict_pipeline', False, 300*1024*1024)):
            adapter = importlib.import_module(cohort+'.aggregation.gait')
            shared = importlib.import_module('dare_wearables.aggregation.gait')
            with patch.object(shared, 'aggregate_gait', return_value=(pd.DataFrame(), pd.DataFrame())) as process:
                adapter.aggregate_gait('synthetic')
                self.assertEqual(process.call_args.kwargs['allow_legacy_hours'], legacy)
            lower_back = importlib.import_module(cohort+'.lower_back.pipeline')
            self.assertEqual(inspect.signature(lower_back.main).parameters['min_size_bytes'].default, threshold)
            paths = importlib.import_module(cohort+'.common.paths')
            self.assertEqual(paths.package_root(), SRC/cohort)
            self.assertEqual(paths.project_root(), ROOT)

    def test_shared_sleep_aggregation_uses_explicit_device_locations(self):
        from dare_wearables.aggregation.sleep import aggregate_sleep
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sensor in ('Empatica', 'GENEActiv'):
                folder = root/'900001'/'T0'/sensor/'sleep_circadian'
                folder.mkdir(parents=True)
                pd.DataFrame({'guider_source': ['HDCZA'], 'spt_found': [True], 'TST': ['0 days 06:00:00']}).to_csv(folder/'sleep_output_all_guiders.csv', index=False)
            bo = importlib.import_module('fallspredict_gp_pipeline.aggregation.sleep').aggregate_sleep(root)
            ra = importlib.import_module('fallspredict_pipeline.aggregation.sleep').aggregate_sleep(root)
            pd.testing.assert_frame_equal(bo, aggregate_sleep(root, sleep_subdir=Path('Empatica/sleep_circadian')))
            pd.testing.assert_frame_equal(ra, aggregate_sleep(root, sleep_subdir=Path('GENEActiv/sleep_circadian')))


if __name__ == '__main__':
    unittest.main()
