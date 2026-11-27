"""Saved-interval coverage contracts using invented recordings only."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts import summarize_joint_wear as wear


class JointWearTests(unittest.TestCase):
    def fixture(self, root):
        emp, run = root / "empatica", root / "mcroberts"
        emp.mkdir()
        folder = run / "silver/900001/T0/McRoberts/nonwear"
        folder.mkdir(parents=True)
        start = pd.Timestamp("2024-01-01")
        hour = pd.Timedelta(hours=1)
        row = dict(subject="900001", visit="T0", recording_start=str(start),
                   protocol_end=str(start + 168*hour), protocol_duration_hours=168,
                   not_masked_hours=166, not_masked_percent=166/168*100,
                   detected_nonwear_hours=2, recording_gap_hours=0, merge_padding_hours=0,
                   missing_recording_hours=0, unassessed_not_masked_hours=0,
                   temperature_alignment_warnings=0, source_short_recording_gaps_1_to_60s_hours=0)
        pd.DataFrame([row]).to_csv(emp / "subject_summary.csv", index=False)
        pd.DataFrame([dict(subject="900001", visit="T0", category="detected_nonwear",
                           start=str(start + 21*hour), end=str(start + 23*hour), duration_hours=2)]
                     ).to_csv(emp / "intervals.csv", index=False)
        (run / "run_status.json").write_text(json.dumps(dict(status="completed", recordings=[
            dict(subject="900001", visit="T0", status="completed")])))
        count = 167*3600*100
        summary = dict(start=str(start-hour), last_sample=str(start+167*hour-pd.Timedelta(milliseconds=10)),
                       sample_rate_hz=100, observed_hours=167, revised_nonwear_hours=2)
        (folder / "nonwear_summary.json").write_text(json.dumps(summary))
        (folder / "timestamp_quality.json").write_text(json.dumps(dict(
            analysis=dict(n_samples=count), streams=[dict(stream="a", retained_samples=count+100)])))
        for name, first, last in [("final_nonwear_intervals.csv", 20, 22),
                                  ("time_invalid_intervals.csv", 10, 11)]:
            pd.DataFrame([dict(start=str(start+first*hour), end=str(start+last*hour))]
                         ).to_csv(folder / name, index=False)
        return emp, run, root / "report"

    def test_joint_coverage_uses_exact_shared_time_and_excludes_missing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            emp, run, output = self.fixture(Path(tmp))
            before = {p: p.read_bytes() for tree in (emp, run) for p in tree.rglob("*") if p.is_file()}
            with redirect_stdout(io.StringIO()):
                wear.analyze(emp, run, output, plots=False)
            result = pd.read_csv(output / "subject_summary.csv", dtype={"subject": str}).iloc[0]
            self.assertEqual(result.subject, "900001")
            self.assertEqual(result.mcroberts_wear_hours, 164)
            self.assertEqual(result.empatica_wear_hours, 166)
            self.assertEqual(result.both_wear_hours, 163)
            self.assertEqual(result.mcroberts_time_invalid_hours, 1)
            self.assertEqual(result.mcroberts_nonwear_hours, 2)
            self.assertEqual(result.mcroberts_cap_censored_tail_hours, 1)
            self.assertEqual(result.both_cap_censored_potential_hours, 1)
            self.assertAlmostEqual(result.both_wear_percent, 163/168*100)
            daily = pd.read_csv(output / "daily_summary.csv")
            self.assertEqual(len(daily), 7)
            self.assertEqual(daily.both_wear_hours.sum(), 163)
            states = pd.read_csv(output / "coverage_states.csv")
            self.assertEqual(states.duration_hours.sum(), 168)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertEqual(json.loads((output / "validation.json").read_text())["n_subjects"], 1)
            self.assertIn("1 matched participants", (output / "report.html").read_text())
            self.assertNotIn("wear_coverage.png", (output / "report.html").read_text())
            self.assertNotIn("Excel workbook", (output / "report.html").read_text())

    def test_duplicate_and_unmatched_cohorts_are_rejected_without_outputs(self):
        for mode in ("duplicate", "unmatched"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                emp, run, output = self.fixture(Path(tmp))
                frame = pd.read_csv(emp / "subject_summary.csv", dtype={"subject": str})
                if mode == "duplicate":
                    frame = pd.concat([frame, frame], ignore_index=True)
                else:
                    frame["subject"] = "900002"
                frame.to_csv(emp / "subject_summary.csv", index=False)
                with self.assertRaises(ValueError):
                    wear.analyze(emp, run, output, plots=False)
                self.assertFalse(output.exists())

    def test_inconsistent_source_totals_and_invalid_intervals_are_rejected(self):
        for mode in ("totals", "interval"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                emp, run, output = self.fixture(Path(tmp))
                if mode == "totals":
                    frame = pd.read_csv(emp / "subject_summary.csv")
                    frame["not_masked_hours"] = 168
                    frame.to_csv(emp / "subject_summary.csv", index=False)
                else:
                    frame = pd.read_csv(emp / "intervals.csv")
                    frame["end"] = frame.start
                    frame.to_csv(emp / "intervals.csv", index=False)
                with self.assertRaises(ValueError):
                    wear.analyze(emp, run, output, plots=False)
                self.assertFalse(output.exists())

    def test_sources_and_existing_reports_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            emp, run, output = self.fixture(Path(tmp))
            for destination in (emp, run, output):
                destination.mkdir(exist_ok=True)
                with self.assertRaises(FileExistsError):
                    wear.analyze(emp, run, destination, plots=False)
            with self.assertRaises(ValueError):
                wear.analyze(emp, run, run / "new_report", plots=False)

    def test_half_open_intervals_and_partition_handle_clipping_and_adjacent_spans(self):
        self.assertEqual(wear.intersect([[0, 3]], [[3, 6]]), [])
        self.assertEqual(wear.union([[2, 3], [0, 2], [5, 6]]), [[0, 3], [5, 6]])
        self.assertEqual(wear.subtract([[0, 10]], [[-2, 1], [3, 5], [8, 20]]), [[1, 3], [5, 8]])
        self.assertEqual(wear.partition([[-1, 2], [2, 3]], [[2, 5], [5, 7]], 0, 6), [
            ["mcroberts_only", 0, 2], ["both", 2, 3], ["empatica_only", 3, 6]])


if __name__ == "__main__":
    unittest.main()
