"""Assessment windows and completion keys on invented, gap-separated samples."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from dare_wearables.lower_back.nonwear import assessment
from dare_wearables.lower_back.nonwear.audit import audit_signature, output_record
from dare_wearables.lower_back.nonwear.ggir2023 import NonwearOptions
from dare_wearables.lower_back.nonwear.respiration import RespirationOptions
from dare_wearables.lower_back.preprocessing import preprocess_lowerback_file
from fallspredict_gp_pipeline.config import LowerBackConfig


class AssessmentWindowTests(unittest.TestCase):
    def test_serial_and_parallel_assessments_honor_elapsed_cap_and_legacy_default(self):
        cases = ((1., 1.5, 6000), (168., 169.5, 6000), (None, 1.5, 12000))
        for workers in (1, 2):
            for cap, second_hour, expected_samples in cases:
                with self.subTest(workers=workers, cap=cap), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    source = root / "bronze/900001/T0/McRoberts/invented.omx"
                    source.parent.mkdir(parents=True)
                    source.write_bytes(b"Invented recording; raw reader is mocked.")
                    start = pd.Timestamp("2024-01-01")
                    times = pd.date_range(start, periods=6000, freq="10ms").append(
                        pd.date_range(start + pd.Timedelta(hours=second_hour), periods=6000, freq="10ms"))
                    raw = pd.DataFrame({"datetime": times, "ax": 0., "ay": 0., "az": 1.,
                                        "gx": 0., "gy": 0., "gz": 0.})
                    config = LowerBackConfig(bronze_root=root / "bronze", silver_root=root / "reports",
                        visits=["T0"], min_size_bytes=0, protocol_hours=cap,
                        nonwear=NonwearOptions(window_seconds=1800., step_seconds=60.),
                        respiration=RespirationOptions(enabled=True, window_seconds=30.))
                    # Exercise the actual parallel submission/worker path in-process
                    # so the invented raw reader is available to the worker too.
                    with patch("dare_wearables.lower_back.preprocessing.read_dp7_as_dataframe_fast",
                               side_effect=lambda *a, **kw: raw.copy()) as reader, \
                         patch("dare_wearables.lower_back.preprocessing.autocalibrate",
                               side_effect=lambda frame, **kw: {"df": frame}), \
                         patch.object(assessment, "ProcessPoolExecutor",
                                      side_effect=lambda **kw: ThreadPoolExecutor(max_workers=kw["max_workers"])), \
                         redirect_stdout(io.StringIO()):
                        expected = preprocess_lowerback_file(str(source), verbose=False,
                            protocol_hours=cap, respiration=config.respiration, nonwear=config.nonwear)
                        run = assessment.run_assessment(config, workers=workers)
                        folder = run / "900001/T0/McRoberts/nonwear"
                        summary = json.loads((folder / "nonwear_summary.json").read_text())
                        self.assertEqual(summary["n_samples"], expected_samples)
                        self.assertEqual(summary["last_sample"], str(expected["df"].datetime.iloc[-1]))
                        self.assertEqual(summary["revised_nonwear_hours"],
                                         expected["respiration"]["summary"]["revised_nonwear_hours"])
                        self.assertEqual(summary["protocol_hours"], cap)
                        self.assertEqual(summary["provenance"]["protocol_hours"], cap)
                        key, _ = audit_signature(source, config.respiration, config.nonwear,
                                                 config.timestamps, protocol_hours=cap)
                        record = output_record(folder, config.respiration)
                        self.assertTrue(record.reusable(key))
                        other_cap = 2. if cap is None else None
                        other_key, _ = audit_signature(source, config.respiration, config.nonwear,
                                                       config.timestamps, protocol_hours=other_cap)
                        self.assertFalse(record.reusable(other_key))
                        calls = reader.call_count
                        assessment.run_assessment(config, resume=run, workers=workers)
                        self.assertEqual(reader.call_count, calls)
                        with self.assertRaisesRegex(ValueError, "Configuration"):
                            assessment.run_assessment(replace(config, protocol_hours=other_cap),
                                                      resume=run, workers=workers)


if __name__ == "__main__":
    unittest.main()
