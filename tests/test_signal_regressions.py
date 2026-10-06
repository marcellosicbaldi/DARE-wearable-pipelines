"""Synthetic regressions for recording alignment and participant selection."""
import importlib
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import pandas as pd

PACKAGES = ('fallspredict_gp_pipeline', 'fallspredict_pipeline')


def packet(sensor, second=0, value=100, count=10):
    result = bytearray(512)
    result[1:2] = sensor
    stamp = (24 << 26) | (1 << 22) | (1 << 17) | second
    struct.pack_into('<I', result, 8, stamp)
    struct.pack_into('<H', result, 16, 100)
    struct.pack_into('<b', result, 18, 1)
    struct.pack_into('<H', result, 22, count)
    result[24:24 + count * 6] = np.full((count, 3), value, dtype='<i2').tobytes()
    return result


class SignalRegressions(unittest.TestCase):
    def test_cohort_is_exact_baseline_match_independent_of_order(self):
        frame = pd.DataFrame({'patient_id': ['900001.0', '9000012', '900001'],
                              'redcap_event_name': ['baseline_arm_1', 'baseline_arm_1', 'followup_arm_1'],
                              'parkinsonf_b': [1, 0, 0]})
        for package in PACKAGES:
            fn = importlib.import_module(package + '.lower_back.gait.gait_functions').get_cohort
            for df in (frame, frame.iloc[::-1]):
                self.assertEqual(fn(df, '0900001'), 'PD')
            with self.assertRaises(ValueError):
                fn(frame, '900002')
            with self.assertRaises(ValueError):
                fn(pd.concat([frame, frame.iloc[:1].assign(parkinsonf_b=0)]), '900001')
            with self.assertRaises(ValueError):
                fn(frame, '900001.5')

    def test_reader_aligns_by_time_and_does_not_shift_missing_packets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'synthetic.OMX'
            path.write_bytes(packet(b'a') + packet(b'a', 1) + packet(b'g', 1, 200))
            for package in PACKAGES:
                for suffix in ('.lower_back.io.mcroberts_loader', '.lower_back.gait.gait_functions'):
                    fn = importlib.import_module(package + suffix).read_dp7_as_dataframe_fast
                    df = fn(path, verbose=0)
                    self.assertEqual(df.datetime.iloc[0], pd.Timestamp('2024-01-01 00:00:01'))
                    self.assertEqual(len(df), 10)
            path.write_bytes(packet(b'a'))
            for package in PACKAGES:
                fn = importlib.import_module(package + '.lower_back.io.mcroberts_loader').read_dp7_as_dataframe_fast
                with self.assertRaisesRegex(ValueError, 'stream'):
                    fn(path, verbose=0)

    def test_reader_preserves_axis_rotation_and_rejects_invalid_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'synthetic.OMX'
            path.write_bytes(packet(b'a', value=4096) + packet(b'g', value=655))
            for package in PACKAGES:
                fn = importlib.import_module(package + '.lower_back.io.mcroberts_loader').read_dp7_as_dataframe_fast
                df = fn(path, verbose=0)
                np.testing.assert_allclose(df[['ax', 'ay', 'az']].iloc[0], [-1, 1, 1])
                np.testing.assert_allclose(df[['gx', 'gy', 'gz']].iloc[0], [-10, 10, 10])
                self.assertEqual(df.attrs['unmatched_acc_samples'], 0)
                self.assertEqual(df.attrs['sample_rate_hz'], 100)
            for payload, error in ((packet(b'a') + packet(b'g', 1), 'no aligned samples'),
                                   (packet(b'a')*2 + packet(b'g'), 'strictly increasing')):
                path.write_bytes(payload)
                for package in PACKAGES:
                    fn = importlib.import_module(package + '.lower_back.io.mcroberts_loader').read_dp7_as_dataframe_fast
                    with self.assertRaisesRegex(ValueError, error):
                        fn(path, verbose=0)

    def test_portions_split_ppg_only_gaps_and_trim_overlap(self):
        index = pd.date_range('2024-01-01', periods=2401, freq='100ms')
        acc = pd.DataFrame({'x': 0}, index=index)
        ppg = pd.DataFrame({'ppg': 0}, index=index[(index < index[800]) | (index > index[1600])])
        for package in PACKAGES:
            fn = importlib.import_module(package + '.wrist.heart_rate.pipeline')._split_recording_portions
            aa, pp = fn(acc, ppg, min_good_portion_minutes=0)
            self.assertEqual(len(aa), 2)
            for a, p in zip(aa, pp):
                self.assertEqual(a.index[0], p.index[0])
                self.assertEqual(a.index[-1], p.index[-1])
                self.assertLessEqual(p.index.to_series().diff().max(), pd.Timedelta('100ms'))
            a, p = fn(acc, ppg.iloc[:0], min_good_portion_minutes=0)
            self.assertEqual((a, p), ([], []))


if __name__ == '__main__':
    unittest.main()
