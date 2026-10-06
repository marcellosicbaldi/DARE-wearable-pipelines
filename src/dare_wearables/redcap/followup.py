"""Fall occurrence dates, questionnaire verification, and monthly ascertainment.

Occurrence month and the REDCap event holding a retrospective report are kept
separate. A completed questionnaire is an operational verification marker,
not evidence that a call happened on schedule or that a month was fully observed.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from dare_wearables.redcap.scores import numeric


BASELINE_EVENT = "baseline_arm_1"
MONTH_PATTERN = r"mese_([1-9]|1[0-2])(?:__follow_up)?_arm_1"
FALL_FIELDS = (
    ("date_fall", "domande_caduta_complete"),
    ("date_fall_c2", "domande_seconda_caduta_7955_complete"),
    ("date_fall_c3", "domande_terza_caduta_complete"),
    ("date_fall_c4", "domande_quarta_caduta_complete"),
    ("date_fall_c5", "domande_quinta_caduta_complete"),
)
FALL_COLUMNS = (
    "subject", "record_id", "redcap_event_name", "source_row", "source_event_month",
    "fall_slot", "start_date", "date_field", "completion_field", "completion_field_present",
    "fall_date_raw", "fall_date", "form_status", "verified", "date_status", "fall_month",
    "source_fall_occurred", "source_reported_count", "hour_fall", "fall_description",
    "possible_duplicate", "recorded_in_different_month",
)


def parse_date(value: object) -> pd.Timestamp:
    """Accept REDCap ISO dates or explicit Italian day/month/year dates."""
    if pd.isna(value) or not str(value).strip():
        return pd.NaT
    text = str(value).strip()
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", text):
        return pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    # Time-of-day is not relevant to calendar-month assignment.
    return pd.to_datetime(text[:10], format="%Y-%m-%d", errors="coerce")


def relative_month(start_date: pd.Timestamp, fall_date: pd.Timestamp) -> int | None:
    """Month m = [baseline + (m-1) calendar months, baseline + m months)."""
    if pd.isna(start_date) or pd.isna(fall_date):
        return None
    start, date = start_date.normalize(), fall_date.normalize()
    if date < start or date >= start + pd.DateOffset(months=12):
        return None
    elapsed = (date.year - start.year) * 12 + date.month - start.month
    if date < start + pd.DateOffset(months=elapsed):
        elapsed -= 1
    return elapsed + 1


def _number(value: object) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) and np.isfinite(number) and number >= 0 else np.nan


def extract_falls(events: pd.DataFrame, baseline_dates: pd.Series) -> pd.DataFrame:
    """One candidate per reported fall slot; retain undated and invalid records."""
    records = []
    for row_number, (_, row) in enumerate(events.iterrows(), start=1):
        event = str(row["redcap_event_name"]).strip().lower()
        if event == BASELINE_EVENT:
            continue  # Baseline fall history is not a prospective outcome.
        subject = row["subject"]
        start = baseline_dates.get(subject, pd.NaT)
        match = re.fullmatch(MONTH_PATTERN, event)
        event_month = int(match[1]) if match else None
        occurred = _number(row.get("cadute_mese", np.nan))
        count = _number(row.get("falls_encounter", np.nan))
        if not np.isnan(count) and count % 1:
            count = np.nan
        for slot, (date_field, completion_field) in enumerate(FALL_FIELDS, start=1):
            raw_date = row.get(date_field, "")
            has_date = pd.notna(raw_date) and bool(str(raw_date).strip())
            status = _number(row.get(completion_field, np.nan))
            if status not in (0, 1, 2):
                status = np.nan
            # The first form is also complete for no-fall calls. Completion alone
            # must never create a first fall on those rows.
            reported = has_date or count >= slot or (slot == 1 and occurred == 1)
            reported = reported or (slot > 1 and status in (1, 2))
            if not reported:
                continue
            date = parse_date(raw_date)
            month = relative_month(start, date)
            if not has_date:
                date_status = "missing_fall_date"
            elif pd.isna(date):
                date_status = "invalid_fall_date"
            elif pd.isna(start):
                date_status = "missing_baseline_date"
            elif date < start:
                date_status = "before_baseline"
            elif month is None:
                date_status = "outside_12_months"
            else:
                date_status = "in_window"
            suffix = "" if slot == 1 else f"_c{slot}"
            records.append({
                "subject": subject, "record_id": row["record_id"], "redcap_event_name": row["redcap_event_name"],
                "source_row": row_number, "source_event_month": event_month, "fall_slot": slot,
                "start_date": start, "date_field": date_field, "completion_field": completion_field,
                "completion_field_present": int(completion_field in events.columns),
                "fall_date_raw": raw_date, "fall_date": date, "form_status": status,
                "verified": int(status == 2) if pd.notna(status) else pd.NA,
                "date_status": date_status, "fall_month": month,
                "source_fall_occurred": occurred, "source_reported_count": count,
                "hour_fall": row.get(f"hour_fall{suffix}", ""),
                "fall_description": row.get(f"fall_description{suffix}", ""),
            })
    falls = pd.DataFrame(records, columns=FALL_COLUMNS)
    for column in ("verified", "form_status", "fall_month", "source_event_month"):
        falls[column] = pd.to_numeric(falls[column], errors="coerce").astype("Int64")
    falls["possible_duplicate"] = (
        falls["fall_date"].notna() & falls.duplicated(["subject", "fall_date"], keep=False)
    ).astype(int)
    falls["recorded_in_different_month"] = falls["fall_month"].ne(falls["source_event_month"]).astype("Int64")
    for column in ("start_date", "fall_date"):
        falls[column] = pd.to_datetime(falls[column]).dt.strftime("%Y-%m-%d")
    return falls.sort_values(["subject", "source_row", "fall_slot"]).reset_index(drop=True)


def extract_followup(
    events: pd.DataFrame, baseline_dates: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return event summaries, subject-wide data, individual falls, calendar months."""
    event = events["redcap_event_name"].str.strip().str.lower()
    month = event.str.extract(rf"^{MONTH_PATTERN}$", expand=False)
    selected = events.loc[month.notna()].copy()
    selected["month"] = month.dropna().astype(int)
    if selected.duplicated(["subject", "month"]).any():
        raise ValueError("Duplicate subject/month rows; resolve repeated instruments before summarizing.")
    monthly = selected[["subject", "record_id", "redcap_event_name", "month"]].copy()
    monthly["fall_occurred"] = numeric(selected, ["cadute_mese"])["cadute_mese"].where(lambda s: s.isin([0, 1]))
    monthly["n_falls_reported"] = numeric(selected, ["falls_encounter"])["falls_encounter"].where(lambda s: s.mod(1).eq(0))
    monthly["fall_response_observed"] = monthly["fall_occurred"].notna().astype(int)
    monthly["fall_count_conflict"] = (
        (monthly["fall_occurred"].eq(0) & monthly["n_falls_reported"].gt(0))
        | (monthly["fall_occurred"].eq(1) & monthly["n_falls_reported"].eq(0))
    ).astype(int)
    status = numeric(selected, ["domande_caduta_complete"])["domande_caduta_complete"].where(lambda s: s.isin([0, 1, 2]))
    monthly["routine_form_status"] = status.astype("Int64")
    monthly["routine_call_verified"] = (
        status.eq(2) & monthly["fall_occurred"].notna()
    ).astype("Int64").where(status.notna())
    monthly["n_reported_falls_beyond_slots"] = (monthly["n_falls_reported"] - len(FALL_FIELDS)).clip(lower=0)
    monthly = monthly.sort_values(["subject", "month"]).reset_index(drop=True)
    falls = extract_falls(events, baseline_dates)

    index = pd.MultiIndex.from_product([baseline_dates.index, range(1, 13)], names=["subject", "month"])
    calendar = pd.DataFrame(index=index)
    routine = monthly.set_index(["subject", "month"])
    calendar["routine_event_present"] = calendar.index.isin(routine.index).astype(int)
    for column in ("fall_occurred", "n_falls_reported", "routine_form_status", "routine_call_verified", "fall_count_conflict"):
        calendar[column] = routine[column].reindex(index)
    allocated = falls.loc[falls["fall_month"].notna()]
    selections = {
        "n_dated_falls": allocated,
        "n_verified_falls": allocated.loc[allocated["verified"].eq(1).fillna(False)],
        "n_unverified_falls": allocated.loc[allocated["verified"].eq(0).fillna(False)],
        "n_unknown_verification_falls": allocated.loc[allocated["verified"].isna()],
        "n_possible_duplicate_entries": allocated.loc[allocated["possible_duplicate"].eq(1)],
    }
    for column, frame in selections.items():
        counts = frame.groupby(["subject", "fall_month"]).size()
        calendar[column] = counts.reindex(index, fill_value=0).astype(int)
    positive = calendar["n_dated_falls"].gt(0)
    verified_negative = (
        calendar["routine_call_verified"].eq(1).fillna(False)
        & calendar["fall_occurred"].eq(0) & calendar["fall_count_conflict"].eq(0)
    )
    # Recorded-entry counts above are always numeric. Outcomes below are missing
    # when neither a dated fall nor a completed explicit no-fall response exists.
    calendar["n_falls_observed"] = calendar["n_dated_falls"].astype("Int64").where(positive | verified_negative)
    calendar["fall_observed"] = positive.astype("Int64").where(positive | verified_negative)
    calendar["date_event_conflict"] = (positive & calendar["fall_occurred"].eq(0)).astype(int)

    wide_data = {}
    subjects = baseline_dates.index
    for m in range(1, 13):
        rows = routine.xs(m, level="month") if m in routine.index.get_level_values("month") else routine.iloc[:0].droplevel("month")
        wide_data[f"followup_event_m{m:02d}"] = pd.Series(subjects.isin(rows.index).astype(int), index=subjects)
        for col in ("fall_occurred", "n_falls_reported", "fall_count_conflict"):
            wide_data[f"{col}_m{m:02d}"] = rows[col].reindex(subjects)
        for col in ("routine_call_verified", "n_dated_falls", "n_verified_falls", "n_unverified_falls",
                    "n_unknown_verification_falls", "n_possible_duplicate_entries", "n_falls_observed",
                    "fall_observed", "date_event_conflict"):
            wide_data[f"{col}_m{m:02d}"] = calendar.xs(m, level="month")[col].reindex(subjects)
    wide = pd.DataFrame(wide_data, index=subjects)
    wide["n_followup_months_observed"] = wide[[f"fall_occurred_m{m:02d}" for m in range(1, 13)]].notna().sum(axis=1)
    wide["n_falls_reported_total"] = wide[[f"n_falls_reported_m{m:02d}" for m in range(1, 13)]].sum(axis=1, min_count=1)
    wide["n_fall_records_unallocated"] = falls.loc[falls["fall_month"].isna()].groupby("subject").size().reindex(subjects, fill_value=0)
    wide["n_fall_records_verified_total"] = falls.loc[falls["verified"].eq(1).fillna(False)].groupby("subject").size().reindex(subjects, fill_value=0)
    wide.index.name = "subject"
    return monthly, wide.reset_index(), falls, calendar.reset_index()
