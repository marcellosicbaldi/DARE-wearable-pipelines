"""Saved-result inspection must preserve label reasons and exact raw alignment."""
import json
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dare_wearables.common.output_state import _digest, signature
from scripts import inspect_lower_back_nonwear as ni


class InspectionTests(unittest.TestCase):
    def fixture(self, root):
        n, fs = 120000, 100
        raw = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=n, freq="10ms"),
                            "ax": 0., "ay": 0., "az": 1., "gx": 0., "gy": 0., "gz": 0.})
        raw.attrs['sample_rate_hz'] = fs
        source = root/'synthetic.omx'
        source.write_bytes(b'synthetic')
        relative = Path('0186/T0/McRoberts/nonwear')
        original, run = root/'original', root/'duration_20261105_example'
        folder, target = original/relative, run/relative
        folder.mkdir(parents=True)
        target.mkdir(parents=True)
        def bouts(first, stop):
            return pd.DataFrame({'start_sample':[first], 'stop_sample':[stop],
                'start':[raw.datetime.iloc[first]], 'end':[raw.datetime.iloc[stop-1]+pd.Timedelta('10ms')],
                'duration_seconds':[(stop-first)/fs]})
        bouts(0,n).to_csv(folder/'accelerometer_nonwear_intervals.csv', index=False)
        bouts(0,6000).to_csv(folder/'respiration_reassigned_intervals.csv', index=False)
        bouts(6000,9000).to_csv(target/'short_nonwear_reassigned_intervals.csv', index=False)
        bouts(9000,n).to_csv(target/'final_nonwear_intervals.csv', index=False)
        summary = {'n_samples':n, 'sample_rate_hz':fs, 'start':str(raw.datetime.iloc[0]),
                   'last_sample':str(raw.datetime.iloc[-1]), 'timestamp_policy':'quarantine_bad_anchors',
                   'provenance':{'timestamps':{'policy':'quarantine_bad_anchors'}}}
        (folder/'nonwear_summary.json').write_text(json.dumps(summary))
        final_summary = {'n_samples':n, 'sample_rate_hz':fs, 'revised_nonwear_hours':1110/3600}
        (target/'nonwear_summary.json').write_text(json.dumps(final_summary))
        source_outputs = {p.name:_digest(p) for p in folder.iterdir()}
        (folder/'respiration_manifest.json').write_text(json.dumps({'outputs':source_outputs}))
        item = {'subject':'0186', 'visit':'T0', 'sensor':'McRoberts', 'reference':{
            'run':str(original), 'source':str(source), 'input_identity':signature({},[source]),
            'completion_sha256':_digest(folder/'respiration_manifest.json')}}
        outputs = {str(relative/p.name):_digest(p) for p in target.iterdir()}
        (run/'duration_filter_manifest.json').write_text(json.dumps({'status':'completed',
            'recordings':[item], 'output_sha256':outputs, 'min_nonwear_bout_seconds':900}))
        return run, raw

    def test_saved_reasons_stay_separate_and_files_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, _ = self.fixture(Path(temporary))
            self.assertEqual(ni.find_saved_run(run.parent), run.resolve())
            saved = ni.load_saved_results(run, '0186')
            self.assertEqual(saved['intervals']['respiration'].duration_seconds.sum(), 60)
            self.assertEqual(saved['intervals']['short_bouts'].duration_seconds.sum(), 30)
            self.assertEqual(saved['intervals']['final_nonwear'].duration_seconds.sum(), 1110)
            self.assertEqual(saved['review_bouts'].category.tolist(), ['respiration','short_bouts','final_nonwear'])
            self.assertEqual(saved['timestamp_policy'], 'quarantine_bad_anchors')
            (run/'0186/T0/McRoberts/nonwear/final_nonwear_intervals.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                ni.load_saved_results(run, '0186')

    def test_raw_loading_uses_recovery_policy_and_checks_all_bout_edges(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, raw = self.fixture(Path(temporary))
            saved = ni.load_saved_results(run, '0186')
            with patch.object(ni,'read_dp7_as_dataframe_fast',return_value=raw) as reader:
                loaded = ni.load_saved_raw(saved)
                self.assertEqual(reader.call_args.kwargs['timestamps'].policy, 'quarantine_bad_anchors')
                np.testing.assert_array_equal(loaded.gy, raw.gy)
            raw.loc[6000,'datetime'] += pd.Timedelta(1,'us')
            with patch.object(ni,'read_dp7_as_dataframe_fast',return_value=raw):
                with self.assertRaisesRegex(ValueError, 'bout boundaries'):
                    ni.load_saved_raw(saved)

    def test_changed_raw_input_is_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, _ = self.fixture(Path(temporary))
            saved = ni.load_saved_results(run, '0186')
            saved['source'].write_bytes(b'changed raw file')
            with patch.object(ni,'read_dp7_as_dataframe_fast') as reader:
                with self.assertRaisesRegex(ValueError, 'OMX changed'):
                    ni.load_saved_raw(saved)
                reader.assert_not_called()

    def test_native_production_assessment_and_historical_selection_are_distinguished(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            old,_=self.fixture(root)
            relative=Path('0186/T0/McRoberts/nonwear')
            native=root/'assessment_20261106_example'
            folder=native/relative
            folder.mkdir(parents=True)
            original=root/'original'/relative
            for name in ['accelerometer_nonwear_intervals.csv','respiration_reassigned_intervals.csv']:
                shutil.copy2(original/name,folder/name)
            for name in ['final_nonwear_intervals.csv','short_nonwear_reassigned_intervals.csv']:
                shutil.copy2(old/relative/name,folder/name)
            summary=json.loads((original/'nonwear_summary.json').read_text())
            summary.update(json.loads((old/relative/'nonwear_summary.json').read_text()))
            summary.update(min_nonwear_bout_seconds=900,baseline_detector={
                'window_seconds':1800,'end_of_recording_correction':False})
            (folder/'nonwear_summary.json').write_text(json.dumps(summary))
            hashes={p.name:_digest(p) for p in folder.iterdir()}
            (folder/'respiration_manifest.json').write_text(json.dumps({'outputs':hashes}))
            old_manifest=json.loads((old/'duration_filter_manifest.json').read_text())
            item=old_manifest['recordings'][0]
            record={k:item[k] for k in ['subject','visit','sensor']}
            record.update(status='completed',source=item['reference']['source'],identity=item['reference']['input_identity'])
            (native/'assessment.json').write_text(json.dumps({'status':'completed','recordings':[record]}))
            selected=ni.find_saved_run(root,subject='0186')
            self.assertEqual(selected,native.resolve())
            saved=ni.load_saved_results(selected,'0186')
            self.assertEqual(saved['baseline_window_seconds'],1800)
            self.assertFalse(saved['end_of_recording_correction'])
            self.assertEqual(saved['intervals']['final_nonwear'].duration_seconds.sum(),1110)
            # A newer unrelated participant must not hide the selected participant's results.
            other=root/'assessment_20261107_other'
            other.mkdir()
            (other/'assessment.json').write_text(json.dumps({'status':'completed','recordings':[
                {'subject':'9999','visit':'T0','status':'completed'}]}))
            self.assertEqual(ni.find_saved_run(root,subject='0186'),native.resolve())


if __name__ == '__main__':
    unittest.main()
