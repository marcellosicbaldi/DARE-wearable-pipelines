"""Full-run inventory/resume/failure contracts; no real recordings are processed."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from dare_wearables.lower_back import batch
from dare_wearables.lower_back.nonwear.ggir2023 import NonwearOptions
from dare_wearables.lower_back.nonwear.respiration import RespirationOptions
from fallspredict_gp_pipeline.config import LowerBackConfig


class FullBatchTests(unittest.TestCase):
    def config(self,root):
        for subject in ('900001','900002'):
            folder=root/'bronze'/subject/'T0'/'McRoberts';folder.mkdir(parents=True)
            (folder/'invented.omx').write_bytes(b'invented OMX, never decoded')
        metadata=root/'metadata.csv';metadata.write_text('invented metadata\n')
        return LowerBackConfig(bronze_root=root/'bronze',silver_root=root/'new_runs',redcap_csv=metadata,
            visits=['T0','T1'],min_size_bytes=1,save_omx_to_parquet=False,protocol_hours=168.,
            nonwear=NonwearOptions(window_seconds=1800.,step_seconds=60.),
            respiration=RespirationOptions(enabled=True,window_seconds=30.,min_nonwear_bout_seconds=900.))

    def test_inventory_is_only_metadata_and_never_starts_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=self.config(Path(tmp))
            with patch('dare_wearables.lower_back.pipeline.run_from_config', side_effect=AssertionError('Unexpected processing')) as process:
                run=batch.prepare(config)
            process.assert_not_called()
            plan=json.loads((run/'run_plan.json').read_text())
            self.assertEqual(len(plan['recordings']),4)
            self.assertEqual(sum(r['status']=='pending' for r in plan['recordings']),2)
            self.assertFalse((run/'silver').exists())
            self.assertEqual(batch._plan(run)[1].nonwear,config.nonwear)

    def test_continues_after_failure_resumes_and_rejects_tampered_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=self.config(Path(tmp));run=batch.prepare(config);processed=[]
            def pipeline(cfg):
                subject=cfg.participant_ids[0];processed.append(subject)
                self.assertEqual(cfg.protocol_hours,168.)
                self.assertEqual(cfg.nonwear.step_seconds,60.)
                if subject=='900001' and processed.count(subject)==1:
                    raise ValueError('Invented first-attempt failure')
                folder=cfg.silver_root/subject/'T0'/'McRoberts'
                for name in ('nonwear/respiration_manifest.json','nonwear/final_nonwear_intervals.csv',
                             'posture/tib_df.csv','posture/tib_qc_df.csv','posture/lying_bouts.csv',
                             'orientation/inversion_episodes.csv','orientation/walking_evidence.csv'):
                    p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('invented\n')
                (folder/'nonwear/nonwear_summary.json').write_text(json.dumps({'observed_hours':1.,'baseline_nonwear_hours':.2,'revised_nonwear_hours':.1}))
                (folder/'orientation/inversion_summary.json').write_text(json.dumps({'inverted_wear_hours':0.,'uncertain_wear_hours':0.}))
                pd.DataFrame(columns=['tib_start','tib_end']).to_csv(folder/'posture/tib_valid_df.csv',index=False)
                gait=folder/'gait';gait.mkdir()
                pd.DataFrame({'day':[1],'hours':[20.],'nonwear_time_minutes':[10.],
                    'processing_status':['completed'],'step_count':[100]}).to_csv(gait/f'{subject}_day_aggregation.csv',index=False)
                (gait/f'{subject}_gait_status.csv').write_text('status\ncompleted\n')
            def worker(command,**kwargs):
                index=int(command[command.index('--record-index')+1])
                try:batch.process_record(run,index);return SimpleNamespace(returncode=0)
                except ValueError:return SimpleNamespace(returncode=1)
            with patch('dare_wearables.lower_back.pipeline.run_from_config',side_effect=pipeline), \
                 patch('dare_wearables.lower_back.nonwear.audit.audit_signature',return_value=('invented',{})), \
                 patch('dare_wearables.lower_back.nonwear.audit.output_record',return_value=SimpleNamespace(reusable=lambda key:True)), \
                 patch.object(batch.subprocess,'run',side_effect=worker):
                self.assertFalse(batch.execute(run))
                self.assertEqual(json.loads((run/'run_status.json').read_text())['status'],'completed_with_failures')
                self.assertTrue(batch.execute(run))
                self.assertEqual(processed,['900001','900002','900001'])
                self.assertTrue(batch.execute(run));self.assertEqual(len(processed),3)
                path=run/'silver/900002/T0/McRoberts/gait/900002_day_aggregation.csv'
                path.write_text('changed output')
                self.assertTrue(batch.execute(run));self.assertEqual(processed[-1],'900002')
            self.assertTrue((run/'aggregation/gait_T0_mean.csv').is_file())
            self.assertTrue((run/'aggregation/lower_back_tib_valid_all.csv').is_file())

    def test_changed_sources_metadata_and_inputs_reject_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=self.config(Path(tmp));run=batch.prepare(config)
            with patch.object(batch,'source_hashes',return_value={'changed':'yes'}):
                with self.assertRaisesRegex(ValueError,'code or environment'):batch.execute(run)
            config.redcap_csv.write_text('changed metadata')
            with self.assertRaisesRegex(ValueError,'metadata changed'):batch.execute(run)
        with tempfile.TemporaryDirectory() as tmp:
            config=self.config(Path(tmp));run=batch.prepare(config)
            (config.bronze_root/'900001/T0/McRoberts/invented.omx').write_bytes(b'changed raw')
            with self.assertRaisesRegex(ValueError,'Input changed'):batch.execute(run)

    def test_ambiguous_recordings_are_skipped_and_outputs_cannot_overlap_bronze(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=self.config(Path(tmp))
            (config.bronze_root/'900001/T0/McRoberts/extra.OMX').write_bytes(b'invented')
            run=batch.prepare(config)
            records=json.loads((run/'run_plan.json').read_text())['recordings']
            self.assertEqual(records[0]['status'],'skipped')
            with self.assertRaises(ValueError):batch.prepare(replace(config,silver_root=config.bronze_root/'out'))


if __name__=='__main__':
    unittest.main()
