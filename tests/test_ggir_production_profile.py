"""Production window duration and evidence-based recording-end decisions."""
from dataclasses import replace
import shutil
import unittest

import numpy as np
import pandas as pd
import polars as pl

from dare_wearables.lower_back.nonwear.ggir2023 import (
    NonwearOptions, detect_nonwear, detector_provenance, _wear_decisions,
)
from scripts.validate_ggir2023_python import run_reference


class ProductionProfileTests(unittest.TestCase):
    def test_experimental_30_minute_window_recovers_removal_missed_by_production(self):
        fs=5
        n=24*3600*fs
        t=np.arange(n)/fs
        moving=.1*np.sin(2*np.pi*.2*t)
        values=np.tile(moving[:,None],(1,3))
        # Record-relative 08:10–08:45: contains a full 30-min grid window but no 60-min window.
        quiet=(t>=8*3600+10*60)&(t<8*3600+45*60)
        values[quiet]=0
        times=pd.date_range('2024-01-01',periods=n,freq='200ms')
        acc=pl.DataFrame({'ts':times.to_numpy(),**{name:values[:,i] for i,name in enumerate('xyz')}})
        production=detect_nonwear(acc,freq=fs)
        experimental=detect_nonwear(acc,freq=fs,options=NonwearOptions(window_seconds=1800.))
        self.assertTrue(production.nonwear_df.is_empty())
        self.assertEqual(production.provenance['window_seconds'],3600)
        self.assertEqual(len(experimental.nonwear_df),1)
        self.assertEqual(experimental.nonwear_df['start'][0],pd.Timestamp('2024-01-01 08:15'))
        self.assertEqual(experimental.nonwear_df['end'][0],pd.Timestamp('2024-01-01 08:45')-pd.Timedelta('200ms'))

    def test_final_wear_is_not_extended_from_an_earlier_nonwear_bout(self):
        scores=np.zeros(200)
        scores[-16:-11]=3 # 75 min of candidate nonwear followed by 165 min of wear.
        old=_wear_decisions(scores,np.zeros(200))
        new=_wear_decisions(scores,np.zeros(200),end_of_recording_correction=False)
        np.testing.assert_array_equal(old[0],new[0])
        self.assertTrue(old[2][-11:].all())
        self.assertFalse(new[2][-11:].any())
        self.assertTrue(new[2][-16:-11].all())

    def test_last_day_download_assumption_is_disabled_but_generic_gap_filling_remains(self):
        scores=np.zeros(200)
        scores[140:146]=3
        scores[154]=3 # Two-hour gap; ratio alone does not qualify for filling.
        old=_wear_decisions(scores,np.zeros(200))
        new=_wear_decisions(scores,np.zeros(200),end_of_recording_correction=False)
        self.assertTrue(old[2][146:154].all())
        self.assertFalse(new[2][146:154].any())
        scores=np.zeros(200)
        scores[60:80]=3
        scores[84:104]=3
        new=_wear_decisions(scores,np.zeros(200),end_of_recording_correction=False)
        self.assertTrue(new[2][80:84].all())

    def test_signal_supported_nonwear_at_the_end_is_preserved(self):
        scores=np.zeros(200)
        scores[-4:]=3
        result=_wear_decisions(scores,np.zeros(200),end_of_recording_correction=False)
        self.assertTrue(result[2][-4:].all())
        self.assertFalse(result[1][-4:].any())

    def test_options_and_provenance_distinguish_production_and_historical_profiles(self):
        options=NonwearOptions.from_mapping({})
        provenance=detector_provenance(options)
        self.assertEqual(provenance['window_seconds'],3600)
        self.assertEqual(provenance['step_seconds'],900)
        self.assertFalse(provenance['end_of_recording_correction'])
        self.assertFalse(provenance['reference_equivalent'])
        self.assertTrue(detector_provenance(replace(options,window_seconds=3600,end_of_recording_correction=True))['reference_equivalent'])
        for bad in (900,2700,'1800',None):
            with self.assertRaises(ValueError):NonwearOptions.from_mapping({'window_seconds':bad})
        with self.assertRaises(ValueError):NonwearOptions.from_mapping({'end_of_recording_correction':'false'})

    @unittest.skipUnless(shutil.which('Rscript'),'R required only for optional upstream parity')
    def test_30_minute_scores_and_original_edge_rules_match_unchanged_r_functions(self):
        for fs in (5.,100.):
            with self.subTest(fs=fs):
                n=int(6*3600*fs+17)
                t=np.arange(n)/fs
                values=np.stack([.08*np.sin(2*np.pi*.19*t+i) for i in range(3)],axis=1)
                values[(t>=3600)&(t<3600+35*60)]=0
                values[t>=5*3600]=0
                times=pd.Timestamp('2024-01-01')+pd.to_timedelta(np.arange(n)/fs,unit='s')
                acc=pl.DataFrame({'ts':times.to_numpy(),**{name:values[:,i] for i,name in enumerate('xyz')}})
                actual=detect_nonwear(acc,freq=fs,options=NonwearOptions(window_seconds=1800.,end_of_recording_correction=True)).window_df.to_pandas()
                expected,_=run_reference(acc,freq=fs,window_seconds=1800)
                pd.testing.assert_frame_equal(actual[expected.columns],expected,check_dtype=False,check_exact=True)


if __name__=='__main__':unittest.main()
