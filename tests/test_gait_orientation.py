"""Physical rotation, gait coordinate conventions and event-count regressions."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

from dare_wearables.lower_back.gait.orientation import (
    ACC_COLUMNS, GYR_COLUMNS, align_gait_sequence, rotate_body_frame,
)
from dare_wearables.lower_back.gait.events import add_bout_event_counts


class GaitOrientationTests(unittest.TestCase):
    def data(self, gravity):
        data = pd.DataFrame(np.tile(gravity, (500, 1)), columns=ACC_COLUMNS)
        data[GYR_COLUMNS] = np.random.default_rng(42).normal(size=(500, 3))
        return data

    def test_upside_down_preserves_forward_and_right_hand_rule(self):
        data = self.data([-9.80665, 0, 0])
        before = data.copy()
        aligned, rotation, info = align_gait_sequence(data, 100)
        np.testing.assert_allclose(rotation, np.diag([-1, -1, 1]), atol=1e-12)
        np.testing.assert_allclose(aligned[ACC_COLUMNS].mean(), [9.80665, 0, 0])
        np.testing.assert_allclose(aligned.gyr_pa, data.gyr_pa)
        a, b = np.array([1., 2., 3.]), np.array([2., -1., 5.])
        np.testing.assert_allclose(np.cross(rotation @ a, rotation @ b), rotation @ np.cross(a, b))
        self.assertTrue(info['vertical_reversed'])
        pd.testing.assert_frame_equal(data, before)

    def test_tilt_alignment_preserves_both_signal_norms(self):
        data = self.data([8., 3., 4.])
        aligned, rotation, info = align_gait_sequence(data, 100)
        self.assertEqual(info['status'], 'aligned')
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)
        for columns in [ACC_COLUMNS, GYR_COLUMNS]:
            np.testing.assert_allclose(np.linalg.norm(data[columns], axis=1),
                                       np.linalg.norm(aligned[columns], axis=1))
        np.testing.assert_allclose(aligned[ACC_COLUMNS].mean().iloc[1:], 0, atol=1e-12)

    def test_reflection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'proper rotation'):
            rotate_body_frame(self.data([9.80665, 0, 0]), -np.eye(3))

    def test_ambiguous_and_short_sequences_are_unchanged(self):
        for data, status in [(self.data([0, 0, 9.80665]), 'ambiguous_forward_axis'),
                             (self.data([9.80665, 0, 0]).iloc[:100], 'insufficient_data')]:
            aligned, rotation, info = align_gait_sequence(data, 100)
            pd.testing.assert_frame_equal(data, aligned)
            np.testing.assert_array_equal(rotation, np.eye(3))
            self.assertEqual(info['status'], status)

    def test_counts_are_events_not_stride_rows_and_turns_have_one_owner(self):
        bouts = pd.DataFrame({'start': [0, 200], 'end': [200, 400], 'n_strides': [2, 2]})
        contacts = pd.DataFrame({'ic': [0, 50, 100, 150, 200, 250, 300, 350, 400, 400]})
        turns = pd.DataFrame({'start': [180, 300], 'end': [220, 350]})
        result = add_bout_event_counts(bouts, contacts, turns)
        self.assertEqual(result.n_raw_initial_contacts.tolist(), [5, 5])
        self.assertEqual(result.n_turns.tolist(), [0, 2])
        self.assertEqual(result.n_strides.tolist(), [2, 2])

    def test_real_extraction_is_invariant_to_upside_down_and_tilted_mounts(self):
        from dare_wearables.lower_back.gait.functions import process_gait_data
        t = np.arange(6000) / 100
        data = pd.DataFrame({
            'timestamp': t, 'acc_x': 9.80665 + 2*np.sin(4*np.pi*t),
            'acc_y': .3*np.sin(2*np.pi*t), 'acc_z': .5*np.cos(4*np.pi*t),
            'gyr_x': 20*np.sin(2*np.pi*t), 'gyr_y': np.cos(2*np.pi*t),
            'gyr_z': 30*np.cos(2*np.pi*t),
        })
        inverted = data.copy()
        inverted[['acc_x', 'acc_y', 'gyr_x', 'gyr_y']] *= -1
        angle = np.deg2rad(25)
        tilt = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
        tilted = data.copy()
        for columns in [['acc_x', 'acc_y', 'acc_z'], ['gyr_x', 'gyr_y', 'gyr_z']]:
            tilted[columns] = data[columns].to_numpy() @ tilt.T
        fixed_gs = pd.DataFrame({'start': [0], 'end': [len(data)]}, index=pd.Index([0], name='gs_id'))
        kwargs = dict(gait_sequences=fixed_gs, participant_height_m=1.7, orientation_mode="sequence")
        upright = process_gait_data(data, **kwargs)
        self.assertIsNotNone(upright['per_wb_params'])
        self.assertFalse(upright['per_wb_params'].empty)
        for candidate in [inverted, tilted]:
            corrected = process_gait_data(candidate, **kwargs)
            pd.testing.assert_frame_equal(upright['initial_contacts'], corrected['initial_contacts'])
            for key in ['stride_duration_s', 'cadence_spm', 'stride_length_m', 'walking_speed_mps',
                        'n_raw_initial_contacts', 'n_turns']:
                np.testing.assert_allclose(upright['per_wb_params'][key], corrected['per_wb_params'][key])
        self.assertTrue((upright['step_counts'] > upright['per_wb_params'].n_strides).all())

    def test_total_height_is_distinct_from_sensor_height(self):
        from dare_wearables.lower_back.gait.functions import get_participant_height, get_sensor_height
        clinical = pd.DataFrame({'patient_id': ['89'], 'redcap_event_name': ['baseline_arm_1'], 'Height': [160.]})
        self.assertEqual(get_participant_height(clinical, '0089'), 1.6)
        self.assertAlmostEqual(get_sensor_height(clinical, subject='0089'), .9856)

    def test_stationary_fixed_sequence_has_no_walking_bouts(self):
        from dare_wearables.lower_back.gait.functions import process_gait_data
        count = 3000
        data = pd.DataFrame({'timestamp': np.arange(count)/100, 'acc_x': 9.80665,
                             'acc_y': 0., 'acc_z': 0., 'gyr_x': 0., 'gyr_y': 0., 'gyr_z': 0.})
        fixed_gs = pd.DataFrame({'start': [0], 'end': [count]}, index=pd.Index([0], name='gs_id'))
        result = process_gait_data(data, gait_sequences=fixed_gs, orientation_mode="sequence")
        # Filter roundoff can produce candidate contacts on a constant signal
        # on some platforms; stride selection still rejects all walking bouts.
        self.assertTrue(result['per_wb_params'] is None or result['per_wb_params'].empty)
        self.assertTrue(result['step_counts'] is None or result['step_counts'].empty)
        self.assertEqual(len(result['orientation']), 1)

    def test_explicit_no_contacts_skips_parameter_extraction(self):
        from dare_wearables.lower_back.gait import functions as gait
        count = 3000
        data = pd.DataFrame({'timestamp': np.arange(count)/100, 'acc_x': 9.80665,
                             'acc_y': 0., 'acc_z': 0., 'gyr_x': 0., 'gyr_y': 0., 'gyr_z': 0.})
        fixed = pd.DataFrame({'start': [0], 'end': [count]}, index=pd.Index([0], name='gs_id'))

        def no_contacts(detector, *args, **kwargs):
            detector.ic_list_ = pd.DataFrame({'ic': pd.Series(dtype='int64')})
            detector.ic_list_.index.name = 'step_id'
            return detector

        with patch.object(gait.IcdIonescu, 'detect', no_contacts), \
                patch.object(gait.CadFromIcDetector, 'calculate',
                             side_effect=AssertionError('No contacts must skip cadence extraction')):
            result = gait.process_gait_data(data, gait_sequences=fixed, orientation_mode='sequence')
        self.assertIsNone(result['per_wb_params'])
        self.assertIsNone(result['step_counts'])
        self.assertTrue(result['initial_contacts'].empty)
        self.assertEqual(len(result['orientation']), 1)

    def test_main_shares_inversion_only_data_between_gait_and_posture(self):
        from dare_wearables.lower_back import pipeline
        times = pd.date_range('2026-01-01', periods=12000, freq='10ms')
        frame = pd.DataFrame({'datetime': times, 'timestamp': np.arange(len(times))/100,
                              'acc_x': -9.80665, 'acc_y': 0., 'acc_z': 0.,
                              'gyr_x': 5., 'gyr_y': 0., 'gyr_z': 0., 'wear': True})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recording = root/'bronze/0089/T0/McRoberts'
            recording.mkdir(parents=True)
            (recording/'synthetic.OMX').write_bytes(b'synthetic')
            clinical = root/'clinical.csv'
            pd.DataFrame({'patient_id': ['89'], 'redcap_event_name': ['baseline_arm_1'],
                          'obs_pt_height': [160.], 'parkinsonf_b': [0]}).to_csv(clinical, index=False)
            with patch.object(pipeline, 'preprocess_lowerback_file', return_value={'df': frame}), \
                    patch('dare_wearables.lower_back.orientation._walking_sequences',
                          return_value=pd.DataFrame({'start':[0,3000,6000], 'end':[2000,5000,8000]})), \
                    patch.object(pipeline, 'apply_orientation_correction', side_effect=AssertionError('Legacy reflection called')), \
                    patch.object(pipeline, 'run_gait_pipeline') as gait, \
                    patch.object(pipeline, 'run_time_in_bed_pipeline') as posture:
                pipeline.main(bronze_root=root/'bronze', silver_root=root/'silver', redcap_csv=clinical,
                              visits=['T0'], min_size_bytes=0, save_omx_to_parquet=False)
            self.assertTrue(gait.call_args.args[0].acc_x.gt(0).all())
            self.assertTrue(posture.call_args.args[0].acc_x.gt(0).all())
            pd.testing.assert_frame_equal(gait.call_args.args[0], posture.call_args.args[0])
            self.assertTrue((root/'silver/0089/T0/McRoberts/orientation/inversion_summary.json').is_file())
            self.assertEqual(gait.call_args.kwargs['participant_height_m'], 1.6)
            self.assertTrue(frame.acc_x.lt(0).all())

    def test_production_extraction_never_calls_tilt_alignment(self):
        from dare_wearables.lower_back.gait.functions import process_gait_data
        count = 3000
        frame = pd.DataFrame({'timestamp': np.arange(count)/100, 'acc_x': 8., 'acc_y': 1.,
                              'acc_z': 4., 'gyr_x': 0., 'gyr_y': 0., 'gyr_z': 0.})
        fixed = pd.DataFrame({'start':[0], 'end':[count]}, index=pd.Index([0],name='gs_id'))
        with patch('dare_wearables.lower_back.gait.functions.align_gait_sequence',
                   side_effect=AssertionError('Unexpected tilt alignment')):
            result = process_gait_data(frame, gait_sequences=fixed)
        self.assertTrue(result['orientation'].empty)


if __name__ == '__main__':
    unittest.main()
