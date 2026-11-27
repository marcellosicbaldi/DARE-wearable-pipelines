"""Mounting correction preserves posture and cannot learn from lying windows."""
import unittest
import numpy as np
import pandas as pd
from dare_wearables.lower_back.posture.orientation import (
    ACC, GYR, proper_rotation_from_gravity, stable_posture_correction,
)
from dare_wearables.lower_back.posture.lying_functions import detect_lying_lowerback


class PostureOrientationTests(unittest.TestCase):
    def frame(self):
        index = pd.date_range('2026-01-01', periods=900, freq='s')
        data = pd.DataFrame(0., index=index, columns=ACC+GYR)
        data.loc[index[:300], 'acc_x'] = 9.80665
        data.loc[index[300:600], 'acc_y'] = 9.80665
        data.loc[index[600:], 'acc_x'] = 9.80665
        data.loc[index[:300], 'gyr_x'] = 30.
        data.loc[index[600:], 'gyr_x'] = 30.
        data['wear'] = True
        return data

    def evidence(self, gravity=(9.80665,0.,0.)):
        index = self.frame().index
        return pd.DataFrame({'start_time': [index[10],index[80],index[650]],
                             'end_time': [index[50],index[120],index[690]],
                             'status': 'aligned', 'gravity_is': gravity[0],
                             'gravity_ml': gravity[1], 'gravity_pa': gravity[2]})

    def test_static_rotation_preserves_real_lying_and_does_not_mutate_wear(self):
        original = self.frame()
        angle = np.deg2rad(25)
        rotation = np.array([[np.cos(angle),0,np.sin(angle)], [0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        mounted = original.copy()
        for columns in [ACC,GYR]: mounted[columns] = original[columns].to_numpy() @ rotation.T
        gravity = rotation @ np.array([9.80665,0.,0.])
        corrected, qc = stable_posture_correction(mounted,self.evidence(gravity))
        self.assertEqual(qc.status.tolist(), ['walking_supported'])
        np.testing.assert_allclose(corrected[ACC+GYR],original[ACC+GYR],atol=1e-12)
        pd.testing.assert_series_equal(corrected.wear,mounted.wear)
        labels,theta,_ = detect_lying_lowerback(corrected)
        self.assertTrue(labels.iloc[330:570].all())
        self.assertTrue((theta.iloc[330:570] > 80).all())
        self.assertFalse(labels.iloc[30:270].any())
        self.assertTrue((mounted.acc_z.iloc[:300] != 0).all())

    def test_upside_down_rotation_is_proper_and_preserves_forward(self):
        rotation=proper_rotation_from_gravity([-9.80665,0,0])
        np.testing.assert_allclose(rotation,np.diag([-1,-1,1]))
        self.assertAlmostEqual(np.linalg.det(rotation),1.)

    def test_lying_does_not_create_a_new_mounting_episode(self):
        _,qc=stable_posture_correction(self.frame(),self.evidence())
        self.assertEqual(len(qc),1)
        self.assertEqual(qc.wear_end.iloc[0],self.frame().index[-1]+pd.Timedelta(seconds=1))

    def test_removal_allows_a_new_mounting_frame(self):
        data=self.frame()
        data.loc[data.index[300:600],'wear']=False
        data.loc[data.index[600:],['acc_x','gyr_x']] *= -1
        first=self.evidence().iloc[:2].copy()
        extra=pd.concat([self.evidence().iloc[:1]]*3,ignore_index=True)
        extra.start_time=[data.index[630],data.index[700],data.index[770]]
        extra.end_time=extra.start_time+pd.Timedelta(seconds=30)
        extra.gravity_is=-9.80665
        # Three independent upright observations in each episode.
        third=first.iloc[:1].copy()
        third.start_time=data.index[160]; third.end_time=data.index[200]
        corrected,qc=stable_posture_correction(data,pd.concat([first,third,extra],ignore_index=True))
        self.assertEqual(qc.status.tolist(),['walking_supported']*2)
        self.assertTrue(corrected.acc_x.iloc[600:].gt(0).all())
        self.assertTrue(corrected.acc_y.iloc[300:600].gt(0).all())
        self.assertFalse(corrected.wear.iloc[300:600].any())

    def test_no_evidence_keeps_identity_and_flags_qc(self):
        data=self.frame()
        corrected,qc=stable_posture_correction(data,self.evidence().iloc[:0])
        pd.testing.assert_frame_equal(corrected,data)
        self.assertEqual(qc.status.tolist(),['insufficient_walking_evidence'])

    def test_opposed_walking_evidence_is_flagged_not_averaged_away(self):
        evidence=pd.concat([self.evidence(),self.evidence((-9.80665,0,0))],ignore_index=True)
        corrected,qc=stable_posture_correction(self.frame(),evidence)
        pd.testing.assert_frame_equal(corrected,self.frame())
        self.assertEqual(qc.status.tolist(),['ambiguous_walking_orientation'])


if __name__=='__main__': unittest.main()
