"""Synthetic verification of labels, raw timing, persistence, and local HTTP API."""
import copy
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from dare_wearables.lower_back.annotation.data import (
    AnnotationConfig, Diary, PAIRS, RawCache, atomic_json, diary_for_recording, write_raw_cache,
)
from dare_wearables.lower_back.annotation.labels import LabelStore, RevisionConflict
from dare_wearables.lower_back.annotation.server import AnnotationApp, make_server


def diary_fixture(path):
    row = {'patient_id':'900001','redcap_event_name':'baseline_arm_1'}
    for day in range(1,8):
        suffix = '' if day == 1 else f'_g{day}'
        row.update({name+suffix:'' for name in ('encounter_ds','no_sensor',*(x for pair in PAIRS for x in pair))})
        row['encounter_ds'+suffix] = f'2024-01-{day:02d}'
    row.update(no_sensor='1',no_sensor_time_2='00:02',no_sensor_time_11='00:04')
    pd.DataFrame([row]).to_csv(path,index=False)
    return row


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.diary_path = self.root/'diary.csv'
        self.diary_row = diary_fixture(self.diary_path)
        n,fs = 100000,100.
        t = np.arange(n,dtype=np.int64)*10_000_000
        t[30000:] += 5_000_000_000  # Time-invalid five-second hole in the raw stream.
        frame = pd.DataFrame({'datetime':pd.Timestamp('2024-01-01')+pd.to_timedelta(t,unit='ns'),
                              **{name:np.zeros(n) for name in ('ax','ay','az','gx','gy','gz')}})
        frame.loc[45234,'gx'] = 123.456789012345
        write_raw_cache(self.root/'cache',frame,source=self.root/'invented.OMX',identity='synthetic-identity',
                        fs=fs,hours=168,policy='strict')
        self.cache = RawCache(self.root/'cache')
        self.diary = Diary(self.diary_path)
        self.imported = diary_for_recording(self.diary,'900001',self.cache)
        self.store = LabelStore(self.root/'labels','900001','T0',self.cache,self.imported)

    def rows(self):
        return copy.deepcopy(self.store.load()['rows'])

    def test_diary_uses_daytime_date_previous_evening_and_only_expected_waist_slots(self):
        row = self.diary_row.copy()
        row.update(no_sensor='3',no_sensor_time_2='14:00',no_sensor_time_11='15:00',
                   no_sensor_time_3='23:00',no_sensor_time_12='07:00',no_sensor_time_6='12:00',
                   no_sensor_time_10='11:00',no_sensor_time_14='11:30')
        pd.DataFrame([row]).to_csv(self.diary_path,index=False)
        data,issues = Diary(self.diary_path).intervals('900001')
        self.assertEqual(len(data),2)
        self.assertEqual(pd.Timestamp(data[0]['start_ns']),pd.Timestamp('2023-12-31 23:00'))
        self.assertEqual(pd.Timestamp(data[0]['end_ns']),pd.Timestamp('2024-01-01 07:00'))
        self.assertEqual(pd.Timestamp(data[1]['start_ns']),pd.Timestamp('2024-01-01 14:00'))
        self.assertEqual(len(issues),2)
        self.assertTrue(any('missing/invalid' in issue for issue in issues))
        self.assertTrue(any('outside declared count' in issue for issue in issues))

    def test_blank_counts_and_equal_times_do_not_become_wear_labels(self):
        row = self.diary_row.copy()
        row.update(no_sensor='',no_sensor_time_2='',no_sensor_time_11='')
        row.update(no_sensor_g2='1',no_sensor_time_2_g2='10:00',no_sensor_time_11_g2='10:00')
        pd.DataFrame([row]).to_csv(self.diary_path,index=False)
        data,issues = Diary(self.diary_path).intervals('900001')
        self.assertEqual(data,[])
        self.assertEqual(issues,['Day 2, period 1: zero duration; excluded.'])

    def test_raw_values_and_sample_boundaries_remain_exact(self):
        self.assertEqual(self.cache.values[45234,3],123.456789012345)
        snapped = self.cache.snap(457.343)
        self.assertEqual(snapped['sample'],45234)
        self.assertEqual(snapped['time'],'2024-01-01T00:07:37.340000')
        self.assertEqual(self.cache.snap(self.cache.duration)['sample'],100000)
        self.assertEqual(self.cache.snap(300,edge='end')['sample'],30000)
        self.assertEqual(self.cache.boundary(30000,edge='end')['seconds'],300)
        self.assertEqual(self.cache.boundary(30000)['seconds'],305)
        for value in (-1,float('nan'),self.cache.duration+1,302):
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.cache.snap(value)

    def test_raw_and_envelope_views_keep_peaks_and_break_lines_at_gaps(self):
        raw = self.cache.view(457,458)
        self.assertEqual(raw['mode'],'raw')
        self.assertEqual(raw['samples'],100)
        self.assertEqual(max(v[3] for v in raw['lo']),123.456789012345)
        envelope = self.cache.view(0,self.cache.duration,width=400)
        self.assertEqual(envelope['mode'],'envelope')
        self.assertEqual(max(v[3] for v in envelope['hi']),123.456789012345)
        self.assertEqual(len(envelope['breaks']),1)
        self.assertEqual(envelope['gaps'],[[300.,305.]])
        empty = self.cache.view(301,304)
        self.assertEqual(empty['mode'],'empty')
        with self.assertRaises(ValueError):self.cache.view(10,1)

    def test_pending_save_and_restart_do_not_create_training_labels(self):
        rows = self.rows()
        rows[0]['start_sample'] += 17
        saved,warning = self.store.save(rows,'Synthetic reviewer',0)
        self.assertIsNone(warning)
        resumed = LabelStore(self.root/'labels','900001','T0',self.cache,self.imported).load()
        self.assertEqual(resumed['rows'],rows)
        self.assertEqual(resumed['revision'],1)
        self.assertEqual(list(csv.DictReader(io.StringIO(self.store.csv_text()))),[])
        self.assertEqual(saved['original_diary'],self.imported['intervals'])

    def test_confirmed_labels_any_duration_and_rejections_export_separately(self):
        rows = self.rows()
        rows[0].update(start_sample=12000,stop_sample=12300,status='reviewed',notes='=invented formula')
        rows.append({'id':'manual_added','label':'wear','status':'reviewed','start_sample':13000,'stop_sample':14000,'notes':''})
        rows.append({'id':'manual_rejected','label':'nonwear','status':'rejected','start_sample':15000,'stop_sample':16000,'notes':''})
        self.store.save(rows,'Synthetic reviewer',0)
        exported = list(csv.DictReader(io.StringIO(self.store.csv_text())))
        self.assertEqual([r['label'] for r in exported],['nonwear','wear'])
        self.assertEqual(float(exported[0]['duration_seconds']),3.)
        self.assertTrue(exported[0]['notes'].startswith("'="))
        self.assertEqual(exported[0]['diary_start'],'2024-01-01T00:02:00')
        self.assertEqual((self.store.folder/'reviewed_intervals.csv').read_text(),self.store.csv_text().replace('\r\n','\n'))

    def test_export_excludes_missing_time_inside_a_reviewed_interval(self):
        rows = self.rows()
        rows[0].update(start_sample=29900,stop_sample=30200,status='reviewed')
        self.store.save(rows,'Synthetic reviewer',0)
        exported = list(csv.DictReader(io.StringIO(self.store.csv_text())))
        self.assertEqual(len(exported),2)
        self.assertEqual(exported[0]['end'],'2024-01-01T00:05:00')
        self.assertEqual(exported[1]['start'],'2024-01-01T00:05:05')
        self.assertEqual(sum(float(r['duration_seconds']) for r in exported),3.)

    def test_reviewers_are_preserved_for_unchanged_labels(self):
        rows = self.rows()
        rows[0]['status'] = 'reviewed'
        self.store.save(rows,'First reviewer',0)
        rows.append({'id':'manual_added','label':'wear','status':'reviewed','start_sample':40000,'stop_sample':41000,'notes':''})
        self.store.save(rows,'Second reviewer',1)
        exported = list(csv.DictReader(io.StringIO(self.store.csv_text())))
        self.assertEqual([row['reviewer'] for row in exported],['First reviewer','Second reviewer'])

    def test_invalid_conflicting_or_ambiguous_labels_are_rejected(self):
        for change in ({'start_sample':-1},{'start_sample':True},{'stop_sample':12000},
                       {'label':'uncertain'},{'status':'unknown'},{'confidence':.8}):
            rows = self.rows()
            rows[0].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                self.store.save(rows,'Synthetic reviewer',0)
        rows = self.rows();rows[0]['status']='reviewed'
        with self.assertRaises(ValueError):self.store.save(rows,'',0)
        rows.append({'id':'manual_overlap','label':'wear','status':'reviewed','start_sample':13000,'stop_sample':14000,'notes':''})
        with self.assertRaisesRegex(ValueError,'overlap'):self.store.save(rows,'Synthetic reviewer',0)
        with self.assertRaisesRegex(ValueError,'original diary'):self.store.save([],'Synthetic reviewer',0)

    def test_revision_conflict_history_atomic_failure_and_source_changes(self):
        rows = self.rows()
        self.store.save(rows,'Synthetic reviewer',0)
        with self.assertRaises(RevisionConflict):self.store.save(rows,'Synthetic reviewer',0)
        rows[0]['notes']='second version'
        self.store.save(rows,'Synthetic reviewer',1)
        self.assertEqual(json.loads((self.store.folder/'history/revision_000001.json').read_text())['revision'],1)
        before = self.store.path.read_bytes()
        with patch('dare_wearables.lower_back.annotation.data.os.replace',side_effect=OSError('invented disk failure')):
            with self.assertRaises(OSError):self.store.save(rows,'Synthetic reviewer',2)
        self.assertEqual(before,self.store.path.read_bytes())
        different = self.imported | {'identity':'changed-diary'}
        with self.assertRaisesRegex(ValueError,'differs'):
            LabelStore(self.root/'labels','900001','T0',self.cache,different).load()

    def test_config_keeps_annotation_outputs_outside_raw_data(self):
        path = self.root/'settings.local.toml'
        text = ('[annotation]\nbronze_root="bronze"\ndiary_csv="diary.csv"\n'
                'output_root="labels"\ncache_root="cached"\nsubject="900001"\n')
        path.write_text(text)
        result = AnnotationConfig.from_toml(path)
        self.assertEqual(result.output_root,(self.root/'labels').resolve())
        path.write_text(text.replace('output_root="labels"','output_root="bronze/labels"'))
        with self.assertRaises(ValueError):AnnotationConfig.from_toml(path)

    def test_http_api_save_resume_snap_export_and_cross_origin_rejection(self):
        (self.root/'bronze/900001/T0/McRoberts').mkdir(parents=True)
        config = AnnotationConfig(self.root/'bronze',self.diary_path,self.root/'labels',self.root/'cache',subject='900001')
        app = AnnotationApp(config,cache_loader=lambda *_:self.cache)
        app.open('900001',background=False)
        server = make_server(app,0)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join,3)
        self.addCleanup(server.shutdown)
        address=f'http://127.0.0.1:{server.server_port}'
        def request(route,data=None,token=True,headers=None):
            h = headers or {}
            if data is not None:
                h = h|{'Content-Type':'application/json'}
                if token:h['X-Annotation-Token']=app.token
            return urlopen(Request(address+route,data=None if data is None else json.dumps(data).encode(),headers=h),timeout=5)
        with request('/') as response:self.assertIn(b'Nonwear review',response.read())
        with request('/api/session') as response:session=json.load(response)
        self.assertEqual(session['status'],'ready')
        with request('/api/view?subject=900001&left=457&right=458') as response:self.assertEqual(json.load(response)['mode'],'raw')
        with request('/api/snap',{'subject':'900001','start':120.001,'end':124}) as response:self.assertEqual(json.load(response)['start']['sample'],12000)
        with self.assertRaises(HTTPError) as error:request('/api/save',{},token=False)
        self.assertEqual(error.exception.code,403)
        with self.assertRaises(HTTPError) as error:request('/api/session',headers={'Host':'untrusted.example'})
        self.assertEqual(error.exception.code,403)
        payload={'subject':'900001','reviewer':'Synthetic reviewer','revision':0,'rows':self.rows()}
        payload['rows'][0]['status']='reviewed'
        with request('/api/save',payload) as response:self.assertEqual(json.load(response)['revision'],1)
        with request('/api/export?subject=900001') as response:self.assertIn('nonwear',response.read().decode())
        with self.assertRaises(HTTPError) as error:request('/api/save',payload)
        self.assertEqual(error.exception.code,409)
        with self.assertRaises(HTTPError):request('/api/view?subject=other&left=0&right=1')


if __name__ == '__main__':
    unittest.main()
