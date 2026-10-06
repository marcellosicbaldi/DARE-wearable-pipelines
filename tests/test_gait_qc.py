"""Regression coverage for mixed legacy Bologna and current daily gait formats."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from gp_pipeline.aggregation.gait import aggregate_gait as aggregate_bologna
from gp_pipeline.aggregation.minimum_data import apply_minimum_observations
from ravenna_pipeline.aggregation.gait import aggregate_gait as aggregate_ravenna


class GaitQcTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def write_days(self, subject, hours, nonwear=None):
        folder = self.root / subject / "T0" / "McRoberts" / "gait"
        folder.mkdir(parents=True, exist_ok=True)
        rows = {"day": range(1, len(hours) + 1), "hours": hours,
                "step_count": [1000.] * len(hours), "wb_all__cadence_spm__avg": [90.] * len(hours)}
        if nonwear is not None:
            rows["nonwear_time_minutes"] = nonwear
        path = folder / f"{subject}_day_aggregation.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_bologna_mixed_formats_use_the_correct_hours_per_file(self):
        legacy = self.write_days("0001", [24., 20., 19., 16.])
        modern = self.write_days("0002", [24., 24., 24., 24.], [0., 420., 480., 600.])
        before = {p: p.read_bytes() for p in (legacy, modern)}
        subjects, days = aggregate_bologna(self.root)
        rows = subjects.set_index("subject")
        self.assertEqual(rows.loc["0001", "gait_n_valid_days"], 3)
        self.assertEqual(rows.loc["0002", "gait_n_valid_days"], 2)
        self.assertEqual(days.loc[days.subject.eq("0001"), "valid_hours"].tolist(), [24., 20., 19., 16.])
        self.assertEqual(days.loc[days.subject.eq("0002"), "valid_hours"].tolist(), [24., 17., 16., 14.])
        self.assertTrue(days.loc[days.subject.eq("0001"), "nonwear_time_minutes"].isna().all())
        self.assertEqual(set(days.valid_hours_source), {"legacy_wear_filtered_hours", "recorded_hours_minus_nonwear"})
        self.assertNotIn("gait_valid_hours_source", subjects)
        masked = apply_minimum_observations(subjects).set_index("subject")
        self.assertEqual(masked.loc["0001", "gait_step_count"], 1000.)
        self.assertTrue(pd.isna(masked.loc["0002", "gait_step_count"]))
        self.assertTrue(pd.isna(masked.loc["0002", "gait_wb_all__cadence_spm__avg"]))
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_legacy_only_bologna_files_are_supported(self):
        self.write_days("0001", [24., 20., 18.])
        subjects, _ = aggregate_bologna(self.root)
        self.assertEqual(subjects.iloc[0].gait_n_valid_days, 3)

    def test_both_cohorts_reject_missing_values_in_existing_qc_columns(self):
        for aggregate in (aggregate_bologna, aggregate_ravenna):
            for hours, nonwear in (([24., 24.], [0., None]), ([24., "invalid"], [0., 0.])):
                with self.subTest(aggregate=aggregate.__module__, hours=hours, nonwear=nonwear):
                    self.write_days("0001", [24.], [0.])
                    self.write_days("0002", hours, nonwear)
                    with self.assertRaisesRegex(ValueError, "0002_day_aggregation.csv.*cannot determine valid days"):
                        aggregate(self.root)

    def test_ravenna_does_not_assume_legacy_bologna_hours(self):
        self.write_days("0001", [24.], [0.])
        self.write_days("0002", [24.])
        with self.assertRaisesRegex(ValueError, "0002_day_aggregation.csv.*nonwear_time_minutes"):
            aggregate_ravenna(self.root)

    def test_modern_formats_retain_genuine_zero_valid_days(self):
        self.write_days("0001", [24., 24.], [1440., 600.])
        for aggregate in (aggregate_bologna, aggregate_ravenna):
            with self.subTest(aggregate=aggregate.__module__):
                subjects, _ = aggregate(self.root)
                self.assertEqual(subjects.iloc[0].gait_n_valid_days, 0)


if __name__ == "__main__":
    unittest.main()
