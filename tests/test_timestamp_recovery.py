"""Conservative quarantine, segmentation and three-state time-quality contracts."""
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl

from dare_wearables.lower_back.io.timestamp_quality import TimestampOptions, quarantine_plan, processing_segments
from dare_wearables.lower_back.io.mcroberts_loader import read_dp7_as_dataframe_fast
from dare_wearables.lower_back.nonwear.ggir2023 import detect_nonwear
from dare_wearables.lower_back.nonwear.respiration import analyze_gyro
from dare_wearables.lower_back.posture.lying_functions import fill_short_nonwear_between_lying, qc_tib_night_wear


def packet(sid, sequence, minute, second, fraction, offset):
    p = bytearray(512)
    p[:2] = b"d"+sid
    struct.pack_into("<H", p, 2, 508)
    struct.pack_into("<I", p, 4, sequence)
    struct.pack_into("<I", p, 8, (24<<26)|(1<<22)|(1<<17)|(minute<<6)|second)
    struct.pack_into("<HhHb", p, 12, fraction, offset, 100, 1)
    p[21] = 0x32
    struct.pack_into("<H", p, 22, 80)
    p[24:504] = np.full((80,3), sequence+1, dtype="<i2").tobytes()
    struct.pack_into("<H", p, 510, (-int(np.frombuffer(p,dtype="<u2").sum()))%65536)
    return bytes(p)


def recording():
    rows = [(0,0,32,43909,80)]
    rows += [(i+1,1,59,64880,2820-80*i) for i in range(35)]
    rows += [(36,1,27,1310,2810)]
    return b"".join(packet(sid,*row) for row in rows for sid in (b"a",b"g"))


class TimestampRecoveryTests(unittest.TestCase):
    def test_only_recognized_anchor_removed_and_retained_samples_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"synthetic.OMX";payload=recording();source.write_bytes(payload)
            with self.assertRaisesRegex(ValueError,"strictly increasing"):
                read_dp7_as_dataframe_fast(source,verbose=0)
            dropped,q=quarantine_plan(source)
            self.assertEqual(len(dropped),70)
            self.assertEqual([e['excluded_samples'] for e in q['events']],[2800,2800])
            result=read_dp7_as_dataframe_fast(source,verbose=0,timestamps=TimestampOptions('quarantine_bad_anchors'))
            self.assertEqual(len(result),160)
            self.assertTrue(result.datetime.is_monotonic_increasing)
            self.assertFalse(result.datetime.duplicated().any())
            self.assertGreater((result.datetime.iloc[80]-result.datetime.iloc[79]).total_seconds(),20)
            np.testing.assert_array_equal(result.ay.iloc[:80],np.full(80,1/4096))
            np.testing.assert_array_equal(result.ay.iloc[80:],np.full(80,37/4096))
            self.assertEqual(source.read_bytes(),payload)
            self.assertFalse(result.attrs['timestamp_quality']['retained_timestamps_changed'])

    def test_bad_checksum_or_unrecognized_pattern_never_silently_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"synthetic.OMX"
            payload=bytearray(recording());payload[2*512+24]^=1;source.write_bytes(payload)
            with self.assertRaisesRegex(ValueError,"checksum"):
                quarantine_plan(source)
            source.write_bytes(packet(b'a',0,0,10,0,0)+packet(b'g',0,0,10,0,0)+
                               packet(b'a',1,0,5,0,0)+packet(b'g',1,0,5,0,0))
            with self.assertRaisesRegex(ValueError,"Unrecognized"):
                quarantine_plan(source)
            with self.assertRaises(ValueError):TimestampOptions('sort').validate()
            with self.assertRaises(ValueError):TimestampOptions.from_mapping({'allow_duplicates':True})

    def test_controls_exactly_equal_between_strict_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"synthetic.OMX"
            source.write_bytes(b''.join(packet(sid,i,0,i+1,0,0) for i in range(3) for sid in (b'a',b'g')))
            strict=read_dp7_as_dataframe_fast(source,verbose=0)
            recovery=read_dp7_as_dataframe_fast(source,verbose=0,timestamps=TimestampOptions('quarantine_bad_anchors'))
            pd.testing.assert_frame_equal(strict,recovery,check_exact=True)
            self.assertFalse(recovery.attrs['timestamp_quality']['events'])
            self.assertIsNone(processing_segments(recovery,100,recovery.attrs['timestamp_quality']))

    def test_ggir_and_respiration_windows_restart_after_quality_break(self):
        # Two segments each contain only one complete 15-minute epoch.
        fs=5.;n=12000;cut=6000
        times=pd.date_range('2024-01-01',periods=n,freq='200ms').to_numpy()
        times[cut:]+=np.timedelta64(30,'s')
        acc=pl.DataFrame({'ts':times,'x':np.zeros(n),'y':np.zeros(n),'z':np.ones(n)})
        result=detect_nonwear(acc,freq=fs,segment_bounds=[[0,cut],[cut,n]])
        windows=result.window_df.to_pandas()
        self.assertEqual(windows.start_sample.tolist(),[0,cut])
        self.assertEqual(windows.stop_sample.tolist(),[4500,10500])
        self.assertEqual(result.provenance['unclassified_tail_samples'],3000)
        gyro=pd.DataFrame({'gx':0.,'gy':.06*np.sin(2*np.pi*19/60*np.arange(n)/fs),'gz':0.},index=times)
        respiration=analyze_gyro(gyro,fs,segment_bounds=[[0,cut],[cut,n]],store_psd=False)
        self.assertFalse(((respiration.windows.start_sample<cut)&(respiration.windows.stop_sample>cut)).any())
        self.assertEqual(len(result.nonwear_df),2)

    def test_missing_time_is_not_nonwear_or_filled_by_lying_correction(self):
        times=pd.date_range('2024-01-01 18:00',periods=180,freq='s')
        wear=pd.Series(True,index=times);observed=pd.Series(True,index=times)
        observed.iloc[60:120]=False
        fixed,bouts,filled=fill_short_nonwear_between_lying(wear,pd.Series(True,index=times),observed_1hz=observed)
        self.assertFalse(fixed.loc[~observed].any())
        self.assertTrue(bouts.empty);self.assertTrue(filled.empty)
        tib=pd.DataFrame({'day_start':[pd.Timestamp('2024-01-01 12:00')],'day_end':[pd.Timestamp('2024-01-02 12:00')],
                          'tib_start':[times[0]],'tib_end':[times[-1]]})
        qc,_=qc_tib_night_wear(tib,wear,observed_1hz=observed)
        self.assertEqual(qc.max_cont_nonwear_h_night.iloc[0],0)
        self.assertEqual(qc.time_invalid_hours_night.iloc[0],1/60)

    def test_tib_exports_unknown_time_separately_from_nonwear(self):
        from dare_wearables.lower_back.pipeline import run_time_in_bed_pipeline
        n=120000
        times=pd.date_range('2024-01-01 18:00',periods=n,freq='10ms')
        frame=pd.DataFrame({'acc_x':0.,'acc_y':0.,'acc_z':9.80665,
                            'gyr_x':0.,'gyr_y':0.,'gyr_z':0.,'wear':True},index=times)
        frame.iloc[:20000,frame.columns.get_loc('gyr_x')]=10.
        frame=frame.loc[(frame.index<times[30000])|(frame.index>=times[33000])].copy()
        frame.attrs['timestamp_quality']={'events':[{'start':str(times[30000]),'end':str(times[33000])}]}
        with tempfile.TemporaryDirectory() as directory:
            run_time_in_bed_pipeline(frame,save_root=directory)
            root=Path(directory)/'nonwear'
            self.assertTrue(pd.read_csv(root/'nonwear_bouts.csv').empty)
            self.assertTrue(pd.read_csv(root/'nonwear_filled_between_lying.csv').empty)
            quality=pd.read_csv(root/'wear_time_quality_1hz.csv')
            self.assertEqual((~quality.time_valid).sum(),30)
            self.assertTrue(quality.loc[~quality.time_valid,'wear'].isna().all())
            self.assertTrue(quality.loc[~quality.time_valid,'wear_after_lying_correction'].isna().all())
