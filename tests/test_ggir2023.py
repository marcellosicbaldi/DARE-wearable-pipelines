"""Parity against the independent comparison bridge and integration contracts."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl

from dare_wearables.lower_back.nonwear.ggir2023 import (
    NonwearOptions, GGIR_COMMIT, GGIR_SOURCES, detect_nonwear,
    detector_provenance, prepare_runtime, _epoch_scores, _wear_decisions,
)
from scripts.validate_ggir2023_python import reference_sources, run_reference
from dare_wearables.lower_back.preprocessing import _build_wear_mask_from_nonwear, preprocess_lowerback_file

REFERENCE = NonwearOptions(window_seconds=3600., end_of_recording_correction=True)


class GGIRContractTests(unittest.TestCase):
    def test_production_default_and_invalid_profiles(self):
        self.assertEqual(NonwearOptions().method, "ggir_2023_matched")
        self.assertEqual(NonwearOptions().window_seconds, 3600.)
        self.assertFalse(NonwearOptions().end_of_recording_correction)
        for value in ("ggir_2023_defaults", "ggir_2023_matched_3axes", "unknown"):
            with self.assertRaises(ValueError):
                NonwearOptions.from_mapping({"method": value})
        with self.assertRaises(ValueError):
            NonwearOptions.from_mapping({"std_threshold": 13})
        with self.assertRaisesRegex(ValueError, "fixes SD"):
            preprocess_lowerback_file("not-read", nonwear_kwargs={"num_axes_required": 3})

    def test_production_needs_no_r_network_or_source_cache(self):
        options = NonwearOptions.from_mapping({"rscript": "nonexistent", "source_cache": "/nonexistent"})
        times = pd.date_range("2024-01-01", periods=4500, freq="200ms")
        acc = pl.DataFrame({"ts": times.to_numpy(), "x": np.zeros(4500), "y": np.zeros(4500), "z": np.ones(4500)})
        with patch("subprocess.run", side_effect=AssertionError("Production called an external program")), \
             patch("urllib.request.urlopen", side_effect=AssertionError("Production accessed the network")):
            prepare_runtime(options)
            provenance = detector_provenance(options)
            result = detect_nonwear(acc, freq=5., options=options)
        self.assertEqual(provenance["implementation"], "python")
        self.assertNotIn("r_version", provenance)
        self.assertEqual(result.window_df["is_final_nonwear"].to_list(), [True])

    def test_strict_range_and_sd_thresholds_and_forward_axis_carry(self):
        # No moving-axis clipping; bypass timestamps to isolate scoring rules.
        medium = 4500
        for amplitude, expected in ((.0129, 3), (.0131, 0)):
            x = np.resize([-amplitude, amplitude], medium)
            score, _ = _epoch_scores([x, x, x], 5., medium, 1)
            self.assertEqual(score[0], expected)
        # Range-only threshold: one extreme at each end, otherwise zero.
        for width, expected in ((.049, 3), (.050, 0), (.051, 0)):
            x = np.zeros(medium)
            x[-2:] = [-width/2, width/2]
            score, _ = _epoch_scores([x, x, x], 5., medium, 1)
            self.assertEqual(score[0], expected)
        x = np.zeros(5 * medium)
        x[4 * medium:] = np.resize([-.1, .1], medium)
        scores, _ = _epoch_scores([x, x, x], 5., medium, 5)
        np.testing.assert_array_equal(scores, [3, 3, 3, 3, 0])

    @unittest.skipUnless(shutil.which("Rscript"), "R is required only for upstream parity")
    def test_randomized_gap_edge_and_iterative_decisions_match_r(self):
        rng = np.random.default_rng(20261102)
        cases = []
        for count in (1, 2, 10, 11, 12, 13, 24, 48, 96, 100, 672):
            for _ in range(20):
                scores = np.repeat(rng.integers(0, 4, count), rng.integers(1, 30, count))[:count]
                clipping = rng.choice([0., .3, .30001, 1.], count)
                cases.append((scores, clipping))
        data = pd.concat([pd.DataFrame({"case": i, "score": score, "clip": clip})
                          for i, (score, clip) in enumerate(cases)], ignore_index=True)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            data.to_csv(folder / "input.csv", index=False)
            script = folder / "decisions.R"
            script.write_text(r'''
args = commandArgs(trailingOnly=TRUE)
source(file.path(args[1], "load_params.R"))
source(file.path(args[1], "g.weardec.R"))
input = read.csv(args[2])
params = load_params()$params_cleaning
output = lapply(split(input, input$case), function(x) {
  result = g.weardec(data.frame(nonwearscore=x$score, clippingscore=x$clip), 2L, 900, params)
  data.frame(case=x$case, initial=as.logical(result$r1), added=as.logical(result$r3),
             final=as.logical(result$r1+result$r3), clipping=as.logical(result$r2))
})
write.csv(do.call(rbind, output), args[3], row.names=FALSE)
''')
            subprocess.run([shutil.which("Rscript"), "--vanilla", str(script), str(reference_sources()),
                            str(folder / "input.csv"), str(folder / "output.csv")],
                           check=True, capture_output=True, text=True)
            expected = pd.read_csv(folder / "output.csv")
        for i, (scores, clipping) in enumerate(cases):
            with self.subTest(case=i, count=len(scores)):
                actual = np.column_stack(_wear_decisions(scores, clipping))
                np.testing.assert_array_equal(actual, expected.loc[expected["case"].eq(i)].iloc[:, 1:].to_numpy())

    @unittest.skipUnless(shutil.which("Rscript"), "R is required only for upstream parity")
    def test_100hz_and_noninteger_target_decimation_match_r(self):
        for fs in (7., 31., 100.):
            with self.subTest(fs=fs):
                n = int(fs * 3600 * 4 + 19)
                time = np.arange(n) / fs
                values = np.zeros((n, 3))
                # Near-SD-threshold motion and changing axis combinations.
                for axis in range(3):
                    moving = ((time // 1800 + axis) % 3) == 0
                    values[:, axis] = .1 * axis + np.where(moving, .08, .0183) * np.sin(2*np.pi*.19*time)
                # Clipping between decimated statistics samples remains visible.
                values[1, 0] = 12.
                times = pd.Timestamp("2024-01-01") + pd.to_timedelta(time, unit="s")
                acc = pl.DataFrame({"ts": times.to_numpy(), **{name: values[:, axis] for axis, name in enumerate("xyz")}})
                actual = detect_nonwear(acc, freq=fs, options=REFERENCE).window_df.to_pandas()
                expected, _ = run_reference(acc, freq=fs)
                pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)

    @unittest.skipUnless(shutil.which("Rscript"), "R is required only for upstream parity")
    def test_floating_point_sd_boundary_matches_r(self):
        # Pairwise NumPy std and R's ordered variance can straddle 13 mg.
        # Test the exact boundary and its adjacent representable amplitudes.
        count = 4500
        amplitude = .013 * np.sqrt((count - 1) / count)
        for fs in (5., 100.):
            values = {}
            for axis, value in zip("xyz", (np.nextafter(amplitude, 0.), amplitude,
                                           np.nextafter(amplitude, np.inf))):
                values[axis] = np.repeat(np.resize([-value, value], count), int(fs / 5))
            n = len(values["x"])
            times = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(n) / fs, unit="s")
            acc = pl.DataFrame({"ts": times.to_numpy(), **values})
            actual = detect_nonwear(acc, freq=fs, options=REFERENCE).window_df.to_pandas()
            expected, _ = run_reference(acc, freq=fs)
            pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)

    def test_short_recording_and_invalid_acceleration(self):
        times = pd.date_range("2024-01-01", periods=10, freq="10ms")
        acc = pl.DataFrame({"ts": times.to_numpy(), "x": np.zeros(10), "y": np.zeros(10), "z": np.ones(10)})
        with patch("subprocess.run") as run:
            result = detect_nonwear(acc)
        run.assert_not_called()
        self.assertTrue(result.nonwear_df.is_empty())
        self.assertTrue(result.window_df.is_empty())
        self.assertEqual(result.provenance["unclassified_tail_samples"], 10)
        with self.assertRaisesRegex(ValueError, "finite"):
            detect_nonwear(acc.with_columns(pl.lit(float("nan")).alias("x")))

    def test_default_preprocessing_selects_ggir_and_preserves_ns_bout_end(self):
        from dare_wearables.lower_back.nonwear.ggir2023 import GGIRNonwearResult
        # Isolate routing and timestamp conversion from the upstream algorithm.
        times = pd.date_range(pd.Timestamp("2024-01-01") + pd.Timedelta(999, "ns"), periods=12500, freq="10ms")
        raw = pd.DataFrame({"datetime": times, "ax": 0., "ay": 0., "az": 1., "gx": 0., "gy": 0., "gz": 0.})
        output = GGIRNonwearResult(pl.DataFrame({"start": times[[0]].to_numpy(), "end": times[[-1]].to_numpy()}),
                                  pl.DataFrame(), {"method": "ggir_2023_matched"})
        with patch("dare_wearables.lower_back.preprocessing.read_dp7_as_dataframe_fast", return_value=raw), \
             patch("dare_wearables.lower_back.preprocessing.autocalibrate", side_effect=lambda df, **kw: {"df": df}), \
             patch("dare_wearables.lower_back.preprocessing.detect_nonwear", return_value=output) as ggir, \
             patch("dare_wearables.lower_back.nonwear.vanhees2013.vanhees2013") as legacy:
            prepared = preprocess_lowerback_file("synthetic", verbose=False)
        ggir.assert_called_once()
        legacy.assert_not_called()
        self.assertFalse(prepared["df"].wear.any())
        self.assertEqual(prepared["nonwear_bouts"].end.iloc[0], times[-1])
        self.assertEqual(prepared["nonwear_options"].method, "ggir_2023_matched")

    def test_real_ggir_and_respiration_reach_production_tib_input(self):
        from dare_wearables.lower_back.nonwear.respiration import RespirationOptions
        from fallspredict_gp_pipeline.config import LowerBackConfig
        from fallspredict_gp_pipeline.lower_back.pipeline import run_from_config
        from dare_wearables.lower_back import pipeline
        n = 120000  # 20 minutes: one complete GGIR epoch and a partial tail.
        times = pd.date_range(pd.Timestamp("2024-01-01 12:00") + pd.Timedelta(999, "ns"), periods=n, freq="10ms")
        raw = pd.DataFrame({"datetime": times, "ax": 0., "ay": 0., "az": 1.,
                            "gx": 0., "gy": .06*np.sin(2*np.pi*(19/60)*np.arange(n)/100), "gz": 0.})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "bronze/900001/T0/McRoberts"
            folder.mkdir(parents=True)
            (folder / "synthetic.OMX").write_bytes(b"synthetic")
            config = LowerBackConfig(bronze_root=root / "bronze", silver_root=root / "silver",
                visits=["T0"], min_size_bytes=1, run_gait=False,
                respiration=RespirationOptions(enabled=True))
            self.assertEqual(config.nonwear.method, "ggir_2023_matched")
            with patch("dare_wearables.lower_back.preprocessing.read_dp7_as_dataframe_fast", return_value=raw), \
                 patch("dare_wearables.lower_back.preprocessing.autocalibrate", side_effect=lambda df, **kw: {"df": df}), \
                 patch.object(pipeline, "apply_orientation_correction", side_effect=lambda df: df), \
                 patch.object(pipeline, "run_time_in_bed_pipeline") as tib:
                run_from_config(config)
            passed = tib.call_args.args[0]
            self.assertFalse(passed.wear_accelerometer.iloc[:90000].any())
            self.assertTrue(passed.wear_accelerometer.iloc[90000:].all())
            self.assertTrue(passed.wear.all())
            self.assertEqual(passed.wear_respiration_override.sum(), 90000)
            output = root / "silver/900001/T0/McRoberts"
            saved = pd.read_parquet(output / "900001_T0_preprocessed.parq")
            np.testing.assert_array_equal(saved.wear, passed.wear)
            import json
            audit = json.loads((output / "nonwear/nonwear_summary.json").read_text())
            self.assertEqual(audit["baseline_method"], "ggir_2023_matched")
            self.assertEqual(audit["baseline_nonwear_hours"], .25)
            self.assertEqual(audit["revised_nonwear_hours"], 0.)

    @unittest.skipUnless(shutil.which("Rscript"), "R is required for upstream parity")
    def test_exact_parity_with_comparison_bridge_including_clipping_and_tail(self):
        from scripts import compare_lower_back_nonwear as reference
        self.assertEqual(GGIR_COMMIT, reference.GGIR_COMMIT)
        self.assertEqual(GGIR_SOURCES, reference.GGIR_SOURCES)
        fs = 5.
        n = int(24 * 3600 * fs + 7)
        t = np.arange(n) / fs
        values = np.tile([0., 0., 1.], (n, 1))
        moving = (t < 2*3600) | ((t >= 7*3600) & (t < 8*3600)) | ((t >= 14*3600) & (t < 21*3600)) | (t >= 23*3600)
        values[moving] += .08 * np.sin(2*np.pi*.13*t[moving, None] + np.array([0., .4, .8]))
        # Only two static axes: exercise the actual >=2-axis production profile.
        one_dynamic = (t >= 9*3600) & (t < 10*3600)
        values[one_dynamic, 0] += .08*np.sin(2*np.pi*.13*t[one_dynamic])
        # Strong clipping in wear must not automatically become nonwear.
        clipping = (t >= 18*3600) & (t < 19*3600)
        values[clipping] = 30*np.sin(2*np.pi*.13*t[clipping, None] + np.array([0., .4, .8]))
        times = pd.date_range(pd.Timestamp("2024-01-01") + pd.Timedelta(999, "ns"), periods=n, freq="200ms")
        acc = pl.DataFrame({"ts": times.to_numpy(), **{name: values[:,i] for i,name in enumerate(("x","y","z"))}})
        sources = reference_sources()
        result = detect_nonwear(acc, freq=fs, options=REFERENCE)
        actual = result.window_df.to_pandas()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            binary = folder / "acc.bin"
            values.astype("<f8").tofile(binary)
            script = folder / "reference.R"
            script.write_text(reference.R_BRIDGE)
            subprocess.run([shutil.which("Rscript"), "--vanilla", str(script), str(sources), str(binary),
                            str(n), str(fs), str(folder), "13", "50"], check=True, capture_output=True, text=True)
            expected = pd.read_csv(folder / "ggir_2023_matched_windows.csv")
        pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)
        self.assertTrue(actual.is_added_nonwear.any())
        self.assertTrue(((actual.nonwear_score == 2) & actual.is_nonwear_window).any())
        self.assertTrue((actual.is_clipping & ~actual.is_final_nonwear).any())
        mask = ~_build_wear_mask_from_nonwear(times, result.nonwear_df.to_pandas())
        expected_mask = np.zeros(n, dtype=bool)
        for row in expected.loc[expected.is_final_nonwear].itertuples():
            expected_mask[int(row.start_sample):int(row.stop_sample)] = True
        np.testing.assert_array_equal(mask, expected_mask)
        self.assertFalse(mask[-7:].any())


if __name__ == "__main__":
    unittest.main()
