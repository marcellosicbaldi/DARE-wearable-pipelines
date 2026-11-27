"""Production contracts and independent mask/audit checks on synthetic signals."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl

from dare_wearables.lower_back.nonwear.ggir2023 import NonwearOptions
from dare_wearables.lower_back.nonwear.respiration import (
    RespirationOptions, analyze_gyro, apply_override, override_frame,
)
from dare_wearables.lower_back.nonwear.audit import output_record, write_audit, audit_signature
from dare_wearables.lower_back.preprocessing import _build_wear_mask_from_nonwear, preprocess_lowerback_file


class IntegrationTests(unittest.TestCase):
    def frame(self, seconds=125, start="2024-01-01 12:00:00", frequency=19 / 60):
        t = np.arange(seconds * 100) / 100
        return pd.DataFrame({"datetime": pd.date_range(start, periods=len(t), freq="10ms"),
                             "gyr_x": 0., "gyr_y": .06 * np.sin(2 * np.pi * frequency * t),
                             "gyr_z": 0., "wear": False})

    def test_band_boundaries_and_daytime(self):
        for rate, expected in ((6, False), (10, False), (19, True), (30, False), (36, False)):
            with self.subTest(brpm=rate):
                frame = self.frame(frequency=rate / 60)
                diag = override_frame(frame, 100, RespirationOptions(enabled=True))
                self.assertEqual(diag["result"].windows.respiration_present.any(), expected)
                self.assertEqual(frame.wear.iloc[:12000].all(), expected)
                self.assertFalse(frame.wear.iloc[12000:].any())
                self.assertFalse(frame.wear_accelerometer.any())
                self.assertIsNone(diag["result"].psd)

    def test_short_and_empty_inputs_preserve_baseline(self):
        for seconds in (0, 20):
            frame = self.frame(seconds=seconds)
            diag = override_frame(frame, 100, RespirationOptions(enabled=True))
            self.assertFalse(frame.wear.any())
            self.assertTrue(diag["result"].windows.empty)
            self.assertAlmostEqual(diag["summary"]["baseline_nonwear_unassessed_hours"], seconds / 3600)

    def test_preprocessing_optional_and_inclusive_end_contract(self):
        from types import SimpleNamespace
        frame = self.frame()
        raw = frame.drop(columns="wear").rename(columns={"gyr_x": "gx", "gyr_y": "gy", "gyr_z": "gz"})
        raw["ax"], raw["ay"], raw["az"] = 0., 0., 1.
        baseline = pd.DataFrame({"start": [raw.datetime.iloc[0]], "end": [raw.datetime.iloc[-1]]})
        nw = SimpleNamespace(nonwear_df=pl.from_pandas(baseline), window_df=pl.DataFrame())
        options = RespirationOptions(enabled=True)
        with patch("dare_wearables.lower_back.preprocessing.read_dp7_as_dataframe_fast", return_value=raw), \
             patch("dare_wearables.lower_back.preprocessing.autocalibrate", side_effect=lambda df, **kw: {"df": df}), \
             patch("dare_wearables.lower_back.nonwear.vanhees2013.vanhees2013", return_value=nw):
            old = preprocess_lowerback_file("synthetic", verbose=False, nonwear=NonwearOptions(method="current"))
            disabled = preprocess_lowerback_file("synthetic", verbose=False, respiration=RespirationOptions(), nonwear=NonwearOptions(method="current"))
            enabled = preprocess_lowerback_file("synthetic", verbose=False, respiration=options, nonwear=NonwearOptions(method="current"))
        pd.testing.assert_frame_equal(old["df"], disabled["df"])
        self.assertNotIn("respiration", disabled)
        np.testing.assert_array_equal(old["df"].wear, enabled["df"].wear_accelerometer)
        self.assertEqual(enabled["df"].wear.sum(), 12000)
        np.testing.assert_array_equal(enabled["df"].wear,
            _build_wear_mask_from_nonwear(enabled["df"].datetime, enabled["nonwear_bouts"]))

    def test_audit_duration_partition_and_completion_integrity(self):
        options = RespirationOptions(enabled=True)
        frame = self.frame(start="2024-01-01 08:59:00")
        diag = override_frame(frame, 100, options)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "synthetic.omx"
            source.write_bytes(b"synthetic")
            folder = root / "audit"
            summary = write_audit({"df": frame, "respiration": diag, "nonwear_options": NonwearOptions(method="current")}, folder, source=source, options=options)
            self.assertAlmostEqual(summary["baseline_nonwear_hours"],
                summary["reassigned_to_wear_hours"] + summary["revised_nonwear_hours"])
            self.assertAlmostEqual(summary["reassigned_09_21_hours"] * 3600, 60.)
            key, _ = audit_signature(source, options, NonwearOptions(method="current"))
            record = output_record(folder, options)
            self.assertTrue(record.reusable(key))
            self.assertFalse(record.reusable(audit_signature(source, replace(options, overlap=.25), NonwearOptions(method="current"))[0]))
            (folder / "final_nonwear_intervals.csv").write_text("corrupted")
            self.assertFalse(record.reusable(key))

    def test_cohort_configs_opt_in_and_validate_unknown_parameters(self):
        from fallspredict_gp_pipeline.config import LowerBackConfig as GP
        from fallspredict_pipeline.config import LowerBackConfig as Ravenna
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            base = f'[lower_back]\nbronze_root = "{temporary}"\nsilver_root = "{temporary}/out"\n'
            for cls in (GP, Ravenna):
                path.write_text(base)
                self.assertFalse(cls.from_toml(path).respiration.enabled)
                self.assertEqual(cls.from_toml(path).nonwear.method, "ggir_2023_matched")
                path.write_text(base + '[lower_back.respiration]\nenabled = true\nband_brpm = [10, 30]\n')
                options = cls.from_toml(path).respiration
                self.assertTrue(options.enabled)
                self.assertEqual(options.settings.band_hz, (10 / 60, 30 / 60))
        with self.assertRaises(ValueError):
            RespirationOptions.from_mapping({"min_band_rsm_dps": .1})
        with self.assertRaises(ValueError):
            RespirationOptions.from_mapping({"enabled": "false"})

class AssessmentTests(unittest.TestCase):
    def test_batch_continues_after_failure_and_resumes_only_valid_outputs(self):
        from fallspredict_gp_pipeline.config import LowerBackConfig
        from dare_wearables.lower_back.nonwear import assessment
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for subject in ("900001", "900002"):
                folder = root / "bronze" / subject / "T0" / "McRoberts"
                folder.mkdir(parents=True)
                (folder / "recording.omx").write_bytes(b"synthetic")
            config = LowerBackConfig(bronze_root=root / "bronze", silver_root=root / "results",
                                     visits=["T0"], min_size_bytes=1,
                                     respiration=RespirationOptions(enabled=True), nonwear=NonwearOptions(method="current"))
            frame_maker = IntegrationTests()
            processed = []
            def process(item, folder, options, expected_sources, nonwear, timestamps=None, *, protocol_hours=None):
                processed.append(item["subject"])
                if item["subject"] == "900001" and processed.count("900001") == 1:
                    raise ValueError("synthetic failure")
                frame = frame_maker.frame()
                diag = override_frame(frame, 100, options)
                return write_audit({"df": frame, "respiration": diag, "nonwear_options": NonwearOptions(method="current"),
                                    "protocol_hours": protocol_hours}, folder,
                                   source=item["source"], options=options)
            real_process = assessment._process
            with patch.object(assessment, "_process", side_effect=process):
                run = assessment.run_assessment(config)
                self.assertEqual(json.loads((run / "assessment.json").read_text())["status"], "completed_with_failures")
                assessment.run_assessment(config, resume=run)
                self.assertEqual(processed, ["900001", "900002", "900001"])
                self.assertEqual(json.loads((run / "assessment.json").read_text())["status"], "completed")
                assessment.run_assessment(config, resume=run)
                self.assertEqual(len(processed), 3)
                changed = replace(config, respiration=replace(config.respiration, overlap=.25))
                with self.assertRaises(ValueError):
                    assessment.run_assessment(changed, resume=run)
                (root / "bronze/900001/T0/McRoberts/recording.omx").write_bytes(b"changed input")
                item = json.loads((run / "assessment.json").read_text())["recordings"][0]
                with self.assertRaisesRegex(ValueError, "Input changed"):
                    real_process(item, root / "invalid", config.respiration, assessment.source_hashes(), config.nonwear)

class PipelinePropagationTests(unittest.TestCase):
    def test_final_wear_reaches_tib_and_saved_parquet_with_audit(self):
        from types import SimpleNamespace
        from fallspredict_gp_pipeline.config import LowerBackConfig
        from fallspredict_gp_pipeline.lower_back.pipeline import run_from_config
        from dare_wearables.lower_back import pipeline
        raw = IntegrationTests().frame().drop(columns="wear").rename(
            columns={"gyr_x": "gx", "gyr_y": "gy", "gyr_z": "gz"})
        raw["ax"], raw["ay"], raw["az"] = 0., 0., 1.
        nw = SimpleNamespace(nonwear_df=pl.DataFrame({"start": [raw.datetime.iloc[0]],
            "end": [raw.datetime.iloc[-1]]}), window_df=pl.DataFrame())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "bronze/900001/T0/McRoberts"
            folder.mkdir(parents=True)
            (folder / "synthetic.OMX").write_bytes(b"synthetic")
            config = LowerBackConfig(bronze_root=root / "bronze", silver_root=root / "silver",
                visits=["T0"], min_size_bytes=1, run_gait=False, run_time_in_bed=True,
                respiration=RespirationOptions(enabled=True), nonwear=NonwearOptions(method="current"))
            with patch("dare_wearables.lower_back.preprocessing.read_dp7_as_dataframe_fast", return_value=raw), \
                 patch("dare_wearables.lower_back.preprocessing.autocalibrate", side_effect=lambda df, **kw: {"df": df}), \
                 patch("dare_wearables.lower_back.nonwear.vanhees2013.vanhees2013", return_value=nw), \
                 patch.object(pipeline, "apply_orientation_correction", side_effect=lambda df: df), \
                 patch.object(pipeline, "run_time_in_bed_pipeline") as tib:
                run_from_config(config)
            passed = tib.call_args.args[0]
            self.assertEqual(passed.wear.sum(), 12000)
            self.assertFalse(passed.wear_accelerometer.any())
            output = root / "silver/900001/T0/McRoberts"
            saved = pd.read_parquet(output / "900001_T0_preprocessed.parq")
            np.testing.assert_array_equal(saved.wear, passed.wear)
            self.assertTrue((output / "nonwear/respiration_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
