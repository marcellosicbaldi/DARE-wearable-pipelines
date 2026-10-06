"""Fall dating and ascertainment regression tests using synthetic participants."""

import unittest

import pandas as pd

from gp_pipeline.redcap.followup import extract_followup, parse_date, relative_month


def event(month, **values):
    name = "mese_6__follow_up_arm_1" if month == 6 else f"mese_{month}_arm_1"
    return {"subject": "0007", "record_id": "a", "redcap_event_name": name, **values}


class FollowupTests(unittest.TestCase):
    def run_rows(self, rows, start="2025-01-15"):
        frame = pd.DataFrame(rows).fillna("")
        dates = pd.Series([parse_date(start)], index=pd.Index(["0007"], name="subject"))
        return extract_followup(frame, dates)

    def test_calendar_month_boundaries_month_end_leap_year_and_day_zero(self):
        for start, date, expected in (
            ("2025-01-15", "2025-01-14", None), ("2025-01-15", "2025-01-15", 1),
            ("2025-01-15", "2025-02-14", 1), ("2025-01-15", "2025-02-15", 2),
            ("2025-01-31", "2025-02-27", 1), ("2025-01-31", "2025-02-28", 2),
            ("2025-01-31", "2025-03-30", 2), ("2025-01-31", "2025-03-31", 3),
            ("2024-02-29", "2025-02-27", 12), ("2024-02-29", "2025-02-28", None),
            ("2025-01-15", "2026-01-14", 12), ("2025-01-15", "2026-01-15", None),
        ):
            with self.subTest(start=start, date=date):
                self.assertEqual(relative_month(parse_date(start), parse_date(date)), expected)
        self.assertEqual(parse_date("28/02/2025"), pd.Timestamp("2025-02-28"))
        self.assertTrue(pd.isna(parse_date("not a date")))

    def test_retrospective_fall_keeps_event_month_and_occurrence_month(self):
        rows = [event(5, cadute_mese="1", falls_encounter="1", date_fall="2025-02-20", domande_caduta_complete="2")]
        monthly, wide, falls, calendar = self.run_rows(rows)
        self.assertEqual(falls.loc[0, "fall_month"], 2)
        self.assertEqual(falls.loc[0, "source_event_month"], 5)
        self.assertEqual(falls.loc[0, "verified"], 1)
        self.assertEqual(falls.loc[0, "recorded_in_different_month"], 1)
        self.assertEqual(wide.loc[0, "n_verified_falls_m02"], 1)
        self.assertEqual(wide.loc[0, "n_verified_falls_m05"], 0)
        c = calendar.set_index("month")
        self.assertEqual(c.loc[2, "fall_observed"], 1)
        self.assertTrue(pd.isna(c.loc[5, "fall_observed"]))
        self.assertEqual(monthly.loc[0, "routine_call_verified"], 1)

    def test_verification_pairs_all_five_slots_by_name_despite_column_order(self):
        row = event(1, cadute_mese="1", falls_encounter="5", domande_sensori_complete="2",
                    domande_quinta_caduta_complete="2", date_fall_c5="2025-02-04",
                    domande_quarta_caduta_complete="", date_fall_c4="2025-02-03",
                    domande_terza_caduta_complete="1", date_fall_c3="2025-02-02",
                    domande_seconda_caduta_7955_complete="0", date_fall_c2="2025-02-01",
                    domande_caduta_complete="2", date_fall="2025-01-31")
        _, _, falls, calendar = self.run_rows([row])
        f = falls.set_index("fall_slot")
        self.assertEqual(f.loc[[1, 2, 3, 5], "verified"].tolist(), [1, 0, 0, 1])
        self.assertTrue(pd.isna(f.loc[4, "verified"]))
        self.assertEqual(f.loc[5, "completion_field"], "domande_quinta_caduta_complete")
        c = calendar.set_index("month").loc[1]
        self.assertEqual(c.n_dated_falls, 5)
        self.assertEqual(c.n_verified_falls, 2)
        self.assertEqual(c.n_unverified_falls, 2)
        self.assertEqual(c.n_unknown_verification_falls, 1)

    def test_month_10_11_12_are_not_month_1_and_lab_form_is_not_phone_call(self):
        rows = [event(10, cadute_mese="0", domande_caduta_complete="2"),
                event(11, cadute_mese="0", domande_caduta_complete="2"),
                event(12, cadute_mese="0", domande_caduta_complete="2"),
                event(6, cadute_mese="0", domande_caduta_complete="0", follow_up_complete="2", followup_complete="2")]
        _, _, falls, calendar = self.run_rows(rows)
        c = calendar.set_index("month")
        self.assertTrue(falls.empty)
        self.assertTrue(pd.isna(c.loc[1, "routine_call_verified"]))
        self.assertTrue(pd.isna(c.loc[1, "fall_observed"]))
        self.assertEqual(c.loc[10:12, "fall_observed"].tolist(), [0, 0, 0])
        self.assertEqual(c.loc[6, "routine_call_verified"], 0)
        self.assertTrue(pd.isna(c.loc[6, "fall_observed"]))

    def test_undated_invalid_outside_and_baseline_records_are_not_misallocated(self):
        rows = [event(1, cadute_mese="1", falls_encounter="2", domande_caduta_complete="2",
                      date_fall_c2="bad date", domande_seconda_caduta_7955_complete="2"),
                event(2, date_fall="2024-12-15", domande_caduta_complete="2"),
                event(3, date_fall="2026-01-15", domande_caduta_complete="2"),
                {**event(1, date_fall="2025-01-20", domande_caduta_complete="2"), "redcap_event_name": "baseline_arm_1"}]
        _, wide, falls, calendar = self.run_rows(rows)
        self.assertEqual(len(falls), 4)
        self.assertEqual(set(falls.date_status), {"missing_fall_date", "invalid_fall_date", "before_baseline", "outside_12_months"})
        self.assertTrue(falls.verified.eq(1).all())
        self.assertEqual(wide.loc[0, "n_fall_records_unallocated"], 4)
        self.assertEqual(calendar.n_dated_falls.sum(), 0)
        self.assertTrue(calendar.fall_observed.isna().all())

    def test_missing_fifth_form_is_unknown_not_unverified_and_counts_are_not_truncated(self):
        _, _, falls, calendar = self.run_rows([
            event(1, cadute_mese="1", falls_encounter="6", date_fall_c5="2025-01-20", domande_caduta_complete="2"),
        ])
        fifth = falls.set_index("fall_slot").loc[5]
        self.assertTrue(pd.isna(fifth.verified))
        self.assertEqual(fifth.completion_field_present, 0)
        monthly, _, _, _ = self.run_rows([event(1, falls_encounter="6")])
        self.assertEqual(monthly.loc[0, "n_reported_falls_beyond_slots"], 1)
        self.assertEqual(calendar.n_unknown_verification_falls.sum(), 1)

    def test_possible_duplicates_are_flagged_without_erasing_same_day_falls(self):
        rows = [event(1, cadute_mese="1", date_fall="2025-01-20", domande_caduta_complete="2"),
                event(2, cadute_mese="1", date_fall="2025-01-20", domande_caduta_complete="2")]
        _, _, falls, calendar = self.run_rows(rows)
        self.assertEqual(len(falls), 2)
        self.assertEqual(falls.possible_duplicate.sum(), 2)
        self.assertEqual(calendar.n_dated_falls.sum(), 2)

    def test_late_fall_can_conflict_with_prior_no_fall_response_without_being_lost(self):
        rows = [event(1, cadute_mese="0", domande_caduta_complete="2"),
                event(2, cadute_mese="1", date_fall="2025-01-20", domande_caduta_complete="1")]
        _, _, falls, calendar = self.run_rows(rows)
        c = calendar.set_index("month")
        self.assertEqual(c.loc[1, "fall_observed"], 1)
        self.assertEqual(c.loc[1, "date_event_conflict"], 1)
        self.assertEqual(c.loc[1, "n_unverified_falls"], 1)
        self.assertEqual(falls.loc[0, "verified"], 0)


if __name__ == "__main__":
    unittest.main()
