"""Map BO/RA longitudinal REDCap exports to the shared T0 non-sensor schema.

The reference CSV is only used for comparison, never to fill source values.
All original event fields are retained separately from the analysis table.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from dare_wearables.redcap.schema import ATC_COLUMNS, DISEASE_COLUMNS, NON_SENSOR_COLUMNS
from dare_wearables.redcap.codebook import ATC_CODES, DISEASE_CODES, SOURCE as CODEBOOK_SOURCE
from dare_wearables.redcap.followup import BASELINE_EVENT, extract_followup, parse_date
from dare_wearables.redcap.fratup import DEFAULT_FRATUP_DIR, add_fratup_arguments, attach_fratup
from dare_wearables.redcap.identity import normalize_subjects
from dare_wearables.redcap.cohorts import COHORT_NAMES, default_redcap_csv, validate_cohort
from dare_wearables.redcap.scores import (
    CESD_REGULAR, CESD_REVERSED, FES_ITEMS, MMSE_ITEMS, PSQI_ITEMS, WFG_ITEMS,
    mmse_corrected_notebook, numeric, psqi_notebook, wfg_notebook,
)


DEFAULT_REDCAP_CSV = default_redcap_csv("BO")
RECRUITMENT = {
    1: "UOC Medicina Interna I", 2: "UOC Medicina Interna II", 3: "Pneumologia",
    4: "Cardiologia", 5: "Medico Specialista Ambulatoriale", 6: "Medico di Medicina Generale",
}
MARITAL_STATUS = {
    1: "Married/cohabiting", 2: "Married/partner not cohabiting", 3: "Separated",
    4: "Divorced", 5: "Widowed", 6: "Single",
}
DROPOUT_REASONS = {
    1: "Rifiuto", 2: "Decesso", 3: "Irraggiungibile", 4: "Compromissione cognitiva",
    5: "Istituzionalizzazione", 6: "Trasferimento",
}


@dataclass
class RedcapResult:
    clinical: pd.DataFrame
    events: pd.DataFrame
    followup_monthly: pd.DataFrame
    followup_wide: pd.DataFrame
    mapping: pd.DataFrame
    falls: pd.DataFrame
    fall_monthly: pd.DataFrame
    codebook_issues: pd.DataFrame
    fratup_inputs: pd.DataFrame
    fratup_import: pd.DataFrame
    fratup_source_paths: tuple[Path, ...]


def load_redcap(path: str | Path) -> pd.DataFrame:
    """Resolve patient IDs within record_id, independent of event row order."""
    raw = pd.read_csv(Path(path).expanduser(), dtype="string", keep_default_na=False)
    required = {"record_id", "patient_id", "redcap_event_name"}
    if missing := required.difference(raw.columns):
        raise ValueError(f"Missing REDCap identity columns: {sorted(missing)}")
    record = raw["record_id"].str.strip().replace("", pd.NA)
    if record.isna().any():
        raise ValueError("Every REDCap row must have record_id; global forward-fill is unsafe.")
    patient = raw["patient_id"].str.strip().replace("", pd.NA)
    known = pd.DataFrame({"record": record[patient.notna()], "subject": normalize_subjects(patient.dropna())})
    if known.groupby("record")["subject"].nunique().gt(1).any():
        raise ValueError("Conflicting patient_id values within a REDCap record_id.")
    if known.groupby("subject")["record"].nunique().gt(1).any():
        raise ValueError("A patient_id belongs to multiple REDCap record_id values.")
    subjects = record.map(known.drop_duplicates("record").set_index("record")["subject"])
    if subjects.isna().any():
        raise ValueError("Some REDCap records have no patient_id on any event.")
    if "subject" in raw:
        # Permit a previously preserved event export as an explicit snapshot
        # input, but revalidate its IDs against the original record mapping.
        if not normalize_subjects(raw["subject"]).eq(subjects).all():
            raise ValueError("Existing subject column disagrees with record_id/patient_id.")
        raw["subject"] = subjects
    else:
        raw.insert(0, "subject", subjects)
    return raw


def _code_columns(frame: pd.DataFrame, family: str) -> list[str]:
    return [c for c in frame if re.fullmatch(rf"{family}(?:_\d+)?f_b", c)]


def _load_codebook(path: str | Path | None) -> dict:
    if path is None:
        return {"source": CODEBOOK_SOURCE, "diseases": DISEASE_CODES, "atc": ATC_CODES}
    book = json.loads(Path(path).expanduser().read_text())
    if not isinstance(book, dict) or not isinstance(book.get("source"), str) or not book["source"].strip():
        raise ValueError("A codebook must identify its source in 'source'.")
    for family, columns in (("diseases", DISEASE_COLUMNS), ("atc", ATC_COLUMNS)):
        if family not in book:
            continue
        mapping = book[family]
        if not isinstance(mapping, dict) or set(mapping) != set(columns):
            raise ValueError(f"Codebook {family} must define all and only the {len(columns)} target columns.")
        for codes in mapping.values():
            if not isinstance(codes, list) or any(type(c) is not int or c < 1 for c in codes):
                raise ValueError(f"Codebook {family} values must be lists of positive integer codes.")
        if not any(mapping.values()):
            raise ValueError(f"Codebook {family} contains no codes.")
    if not any(f in book for f in ("diseases", "atc")):
        raise ValueError("Codebook must contain diseases and/or atc mappings.")
    return book


def audit_codebook(book: dict, baseline: pd.DataFrame) -> pd.DataFrame:
    """Expose overlaps and codes excluded by the supplied analysis dictionary."""
    issues = []
    for family, prefix in (("diseases", "patologies_code"), ("atc", "therapies_code")):
        definitions = book.get(family, {})
        by_codes: dict[tuple[int, ...], list[str]] = {}
        for column, codes in definitions.items():
            if codes:
                by_codes.setdefault(tuple(sorted(codes)), []).append(column)
        for codes, names in by_codes.items():
            if len(names) > 1:
                issues.append({"family": family, "issue": "identical_category_codes", "target_columns": ", ".join(names),
                               "source_code": "", "subject_count": pd.NA,
                               "note": f"Identical {len(codes)}-code mapping retained from {book['source']}; category totals count both. Requires dictionary review."})
        code_frame = numeric(baseline, _code_columns(baseline, prefix))
        known = {code for codes in definitions.values() for code in codes}
        present = {float(code) for code in code_frame.to_numpy().ravel() if pd.notna(code)}
        for code in sorted(present - known):
            issues.append({"family": family, "issue": "unmapped_source_code", "target_columns": "",
                           "source_code": code, "subject_count": int(code_frame.eq(code).any(axis=1).sum()),
                           "note": "Code absent from supplied analysis categories; raw value retained."})
    return pd.DataFrame(issues, columns=["family", "issue", "target_columns", "source_code", "subject_count", "note"])


def process_redcap(
    redcap_csv: str | Path | None = None,
    *,
    cohort: str = "BO",
    codebook_path: str | Path | None = None,
    fratup_dir: str | Path | None = DEFAULT_FRATUP_DIR,
) -> RedcapResult:
    """Extract T0 clinical variables, preserve all visits, and describe every mapping."""
    cohort = validate_cohort(cohort)
    events = load_redcap(default_redcap_csv(cohort) if redcap_csv is None else redcap_csv)
    event = events["redcap_event_name"].str.strip().str.lower()
    baseline = events.loc[event.eq(BASELINE_EVENT)].replace("", pd.NA).set_index("subject")
    if baseline.empty:
        raise ValueError("No baseline_arm_1 rows found.")
    if baseline.index.duplicated().any():
        raise ValueError("Duplicate baseline rows for a subject; refusing a many-to-many merge.")
    baseline = baseline.sort_index()
    out = pd.DataFrame(index=baseline.index)
    out["subject"] = baseline.index
    metadata: dict[str, dict[str, str]] = {}

    def record(target: str, sources: tuple[str, ...] | list[str], rule: str,
               status: str = "mapped") -> bool:
        missing = [c for c in sources if c not in baseline]
        metadata[target] = {"variable": target, "status": "missing_source_columns" if missing else status,
                            "source_fields": ", ".join(sources), "rule": rule,
                            "missing_source_columns": ", ".join(missing)}
        return not missing

    def num(source: str) -> pd.Series:
        return numeric(baseline, [source])[source]

    def direct(target: str, source: str, labels: dict | None = None) -> None:
        record(target, [source], "Source value; negative sentinel -> missing." if labels is None else str(labels))
        values = num(source)
        out[target] = values.map(labels) if labels is not None else values

    record("subject", ["patient_id", "record_id"], "patient_id mapped within record_id; four-digit string.")
    record("group", [], f"Explicit {COHORT_NAMES[cohort]} cohort selection: {cohort}.")
    out["group"] = cohort
    for target, source, labels in (
        ("sex", "patient_gender", {1: "M", 2: "F"}), ("age", "patient_age", None),
        ("education_years", "patient_school", None), ("marital_status", "marital_status", MARITAL_STATUS),
        ("dropout_binario", "dropout_yn", {0: 0, 1: 1}), ("cfs", "cfsf_b", None),
        ("tug_time_s", "test_timed_b", None), ("history_falls", "previous_fallsf_b", {0: 0, 1: 1}),
        ("dizziness", "dizziness_or_unsteadinessf_b", {0: 0, 1: 1}),
        ("living_alone", "living_alonef_b", {0: 0, 1: 1}),
        ("daily_medications", "daily_medications_spf_b", None),
    ):
        direct(target, source, labels)
    for target, source, other, labels in (
        ("recruitment", "patient_recruitment", "rec_oth", RECRUITMENT),
        ("dropout_why", "dropout_r", "dropout_r_oth", DROPOUT_REASONS),
    ):
        record(target, [source, other], f"{labels}; -2 uses {other}.")
        codes = pd.to_numeric(baseline.get(source, pd.Series(index=baseline.index, dtype=float)), errors="coerce")
        out[target] = codes.map(labels).astype("string")
        out.loc[codes.eq(-2), target] = baseline.reindex(columns=[other]).loc[codes.eq(-2), other]
    record("start_date_T0", ["encounter_fof_b"], "Baseline encounter date, ISO YYYY-MM-DD.")
    dates = baseline.reindex(columns=["encounter_fof_b"])["encounter_fof_b"].map(parse_date)
    out["start_date_T0"] = dates.dt.strftime("%Y-%m-%d")
    record("history_falls_injurious", ["previous_fallsf_b", "falls_lesionf_b"],
           "0 when previous_falls=0; otherwise falls_lesion (unknown stays missing).")
    out["history_falls_injurious"] = num("falls_lesionf_b").where(lambda s: s.isin([0, 1]))
    out.loc[out["history_falls"].eq(0), "history_falls_injurious"] = 0
    for target, items in (
        ("mmse", MMSE_ITEMS), ("shortFESI", FES_ITEMS),
        ("adl", tuple(f"physcial_disabilityf_b___{i}" for i in range(1, 7))),
        ("iadl", tuple(f"instrumental_disabilityf_b___{i}" for i in range(1, 9))),
    ):
        record(target, items, f"Sum of {len(items)} items; all items required.")
        out[target] = numeric(baseline, items).sum(axis=1, min_count=len(items))
    record("cesd", [*CESD_REGULAR, *CESD_REVERSED], "Sum(item-1) for 16 regular items + sum(4-item) for 4 reversed; all required.")
    out["cesd"] = (numeric(baseline, CESD_REGULAR) - 1).sum(axis=1, min_count=16) + (4 - numeric(baseline, CESD_REVERSED)).sum(axis=1, min_count=4)
    walks = ("walk_test_andata_b", "walk_test_ritorno_b")
    record("gait_speed", walks, "4 / maximum available positive walk time (metres/second).")
    times = numeric(baseline, walks).where(lambda d: d > 0).max(axis=1)
    out["gait_speed"] = 4 / times
    record("mmse_corrected", ["patient_age", "patient_school"],
           "Cell-6 commented equation: 2.4*log10(93.9-age)*education_years**0.29+22.1; age<93.9. Does not use observed MMSE.",
           "notebook_compatibility")
    out["mmse_corrected"] = mmse_corrected_notebook(out["age"], out["education_years"])
    for target, fields, function in (("psqi", PSQI_ITEMS, psqi_notebook), ("wfg", WFG_ITEMS, wfg_notebook)):
        if record(target, fields, "Compatibility with supplied notebook; see documented scoring limitations.", "notebook_compatibility"):
            out[target] = function(baseline)

    for target, status, reason in (
        ("fratup", "external_disabled", "External CSV import explicitly disabled; score stays missing."),
        ("fallen_6m", "deferred", "Exact six-month outcome deferred by user; all monthly observations exported."),
        ("followup_6m", "deferred", "Exact six-month follow-up rule deferred by user; all monthly observations exported."),
    ):
        record(target, [], reason, status)

    book = _load_codebook(codebook_path)
    historical = codebook_path is None
    for family, prefix, columns, total, count_field in (
        ("diseases", "patologies_code", DISEASE_COLUMNS, "n_morbidities", "patologiesf_b"),
        ("atc", "therapies_code", ATC_COLUMNS, "n_medications", "therapiesf_b"),
    ):
        raw_columns = _code_columns(baseline, prefix)
        codes = numeric(baseline, raw_columns)
        definitions = book.get(family)
        if definitions is None:
            for column in (*columns, total):
                record(column, raw_columns, "Family omitted from the supplied replacement codebook; raw fields retained in events export.", "unresolved_codebook")
            continue
        known_codes = {code for values in definitions.values() for code in values}
        unknown = (codes.notna() & ~codes.isin(known_codes)).any(axis=1)
        observed = codes.notna().any(axis=1) | num(count_field).eq(0)
        if unknown.any() and not historical:
            warnings.warn(f"{family}: {int(unknown.sum())} subjects have unmapped codes; absent indicators and totals stay missing.", stacklevel=2)
        indicators = {}
        for column in columns:
            positive = codes.isin(definitions[column]).any(axis=1)
            indicators[column] = positive.astype(float) if historical else positive.astype(float).where((observed & ~unknown) | positive)
            if not raw_columns:
                indicators[column][:] = np.nan
            record(column, raw_columns or [f"{prefix}f_b"],
                   f"Codes {list(definitions[column])}; source: {book['source']}. "
                   + ("No matching code -> 0 (notebook convention); see codebook_issues export." if historical else "Unknown source codes propagate missing negative indicators."),
                   "notebook_compatibility" if historical else "provided_codebook")
        category_frame = pd.DataFrame(indicators, index=baseline.index)
        category_frame[total] = category_frame.sum(axis=1, min_count=len(columns))
        out = pd.concat([out, category_frame], axis=1)
        record(total, raw_columns or [f"{prefix}f_b"], f"Sum of all {len(columns)} category flags, not number of raw slots. Historical overlapping categories both contribute.",
               "notebook_compatibility" if historical else "provided_codebook")

    out = out.reindex(columns=NON_SENSOR_COLUMNS).reset_index(drop=True)
    fratup = attach_fratup(out, baseline.reset_index()[["subject", "record_id"]], fratup_dir, cohort=cohort)
    out = fratup.clinical
    metadata.update({row["variable"]: row for row in fratup.mapping})
    mapping = pd.DataFrame([metadata[c] for c in out])
    mapping["nonmissing_rows"] = [int(out[c].notna().sum()) for c in out]
    monthly, wide, falls, calendar = extract_followup(events, dates)
    return RedcapResult(out, events, monthly, wide, mapping, falls, calendar,
                        audit_codebook(book, baseline), fratup.inputs, fratup.audit, fratup.source_paths)


def aggregate_redcap(
    redcap_csv: str | Path | None = None,
    *, visit: str = "T0", cohort: str = "BO", codebook_path: str | Path | None = None,
    fratup_dir: str | Path | None = DEFAULT_FRATUP_DIR,
) -> pd.DataFrame:
    """Return clinical + monthly data with subject/visit keys for sensor aggregation."""
    if visit != "T0":
        raise ValueError("The reference clinical schema is T0-only. All follow-up events are exported separately.")
    result = process_redcap(redcap_csv, cohort=cohort, codebook_path=codebook_path, fratup_dir=fratup_dir)
    unresolved = result.mapping["status"].str.startswith("unresolved").sum()
    if unresolved:
        warnings.warn(f"REDCap: {unresolved} columns need a correction formula/current codebook and remain missing. See gp-aggregate-redcap mapping export.", stacklevel=2)
    identical = result.codebook_issues.loc[result.codebook_issues["issue"].eq("identical_category_codes"), "target_columns"]
    if not identical.empty:
        warnings.warn(f"REDCap codebook assigns identical codes to: {'; '.join(identical)}. Values retained; inspect redcap_codebook_issues.csv.", stacklevel=2)
    data = result.clinical.merge(result.followup_wide, on="subject", validate="one_to_one")
    data.insert(1, "visit", visit)
    return data


def merge_redcap_with_sensors(clinical: pd.DataFrame, sensors: pd.DataFrame, *, visit: str = "T0", cohort: str | None = None) -> pd.DataFrame:
    """Outer join with cohort-scoped IDs when group is present or supplied."""
    if cohort is not None:
        validate_cohort(cohort)
    grouped = cohort is not None or "group" in clinical or "group" in sensors
    if grouped and cohort is None:
        groups = set()
        for frame in (clinical, sensors):
            if "group" in frame:
                groups.update(frame["group"].dropna().unique())
        if len(groups) == 1:
            cohort = validate_cohort(next(iter(groups)))
        elif "group" not in clinical or "group" not in sensors:
            raise ValueError("Both tables need group when merging multiple cohorts.")
    keys = ["group", "subject", "visit"] if grouped else ["subject", "visit"]
    frames = []
    for original in (clinical, sensors):
        frame = original.copy()
        if grouped:
            if "group" not in frame:
                frame["group"] = cohort
            if not frame["group"].isin(COHORT_NAMES).all():
                raise ValueError("Invalid or missing cohort group in merge.")
            if cohort is not None and frame["group"].ne(cohort).any():
                raise ValueError(f"Merge expects only {cohort} rows.")
        if "subject" not in frame:
            raise ValueError("Both tables must contain subject.")
        frame["subject"] = normalize_subjects(frame["subject"])
        if "visit" not in frame:
            frame.insert(1, "visit", visit)
        frame["visit"] = frame["visit"].astype("string").str.strip().str.upper()
        if frame["visit"].isna().any() or frame["visit"].ne(visit).any():
            raise ValueError(f"Merge expects only {visit} rows; filter the sensor table to that visit first.")
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate cohort/subject/visit keys; refusing a many-to-many merge.")
        frames.append(frame)
    overlap = set(frames[0]).intersection(frames[1]).difference(keys)
    if overlap:
        raise ValueError(f"Overlapping non-key columns in clinical and sensor tables: {sorted(overlap)}")
    return frames[0].merge(frames[1], on=keys, how="outer", validate="one_to_one").sort_values(keys).reset_index(drop=True)


def compare_reference(clinical: pd.DataFrame, reference_csv: str | Path) -> pd.DataFrame:
    """Aggregate comparison only; never copy participant values from the reference."""
    reference = pd.read_csv(Path(reference_csv).expanduser(), dtype={"subject": "string"})
    if "guider_used" in reference:
        reference = reference.iloc[:, :reference.columns.get_loc("guider_used")]
    if tuple(reference.columns) != NON_SENSOR_COLUMNS:
        raise ValueError("Reference columns before guider_used do not match the saved T0 schema.")
    reference["subject"] = normalize_subjects(reference["subject"])
    if reference["subject"].duplicated().any():
        raise ValueError("Reference has duplicate subjects.")
    left = clinical.set_index("subject")
    right = reference.set_index("subject")
    keys = left.index.intersection(right.index)
    rows = []
    for col in NON_SENSOR_COLUMNS[1:]:
        a, b = left.loc[keys, col], right.loc[keys, col]
        both = a.notna() & b.notna()
        if pd.api.types.is_numeric_dtype(b):
            equal = pd.Series(np.isclose(pd.to_numeric(a, errors="coerce").to_numpy(dtype=float, na_value=np.nan),
                                        pd.to_numeric(b, errors="coerce").to_numpy(dtype=float, na_value=np.nan),
                                        rtol=1e-9, atol=1e-9, equal_nan=True), index=keys)
        else:
            equal = a.astype("string").eq(b.astype("string")).fillna(False) | (a.isna() & b.isna())
        rows.append({"variable": col, "shared_subjects": len(keys), "equal_including_missing": int(equal.sum()),
                     "different_nonmissing": int((both & ~equal).sum()),
                     "source_missing_reference_present": int((a.isna() & b.notna()).sum()),
                     "source_present_reference_missing": int((a.notna() & b.isna()).sum()),
                     "source_only_subjects": len(left.index.difference(right.index)),
                     "reference_only_subjects": len(right.index.difference(left.index))})
    return pd.DataFrame(rows)


def write_redcap_exports(
    redcap_csv: str | Path | None = None, *, output_dir: str | Path, cohort: str = "BO",
    codebook_path: str | Path | None = None, reference_csv: str | Path | None = None,
    sensor_csv: str | Path | None = None,
    fratup_dir: str | Path | None = DEFAULT_FRATUP_DIR,
) -> dict[str, Path]:
    redcap_csv = default_redcap_csv(cohort) if redcap_csv is None else redcap_csv
    result = process_redcap(redcap_csv, cohort=cohort, codebook_path=codebook_path, fratup_dir=fratup_dir)
    tables = {"clinical_T0": result.clinical, "events": result.events,
              "followup_monthly": result.followup_monthly, "followup_wide": result.followup_wide,
              "mapping": result.mapping, "falls": result.falls,
              "fall_monthly": result.fall_monthly, "codebook_issues": result.codebook_issues,
              "fratup_inputs": result.fratup_inputs, "fratup_import": result.fratup_import}
    if reference_csv is not None:
        if cohort != "BO":
            raise ValueError("The May reference comparison is defined for Bologna only.")
        tables["reference_comparison"] = compare_reference(result.clinical, reference_csv)
    if sensor_csv is not None:
        clinical = result.clinical.merge(result.followup_wide, on="subject", validate="one_to_one")
        tables["clinical_sensors_T0"] = merge_redcap_with_sensors(
            clinical, pd.read_csv(Path(sensor_csv).expanduser(), dtype={"subject": "string"}), cohort=cohort,
        )
    folder = Path(output_dir).expanduser()
    paths = {name: folder / f"redcap_{name}.csv" for name in tables}
    inputs = {Path(p).expanduser().resolve() for p in (redcap_csv, reference_csv, sensor_csv, codebook_path) if p is not None}
    inputs.update(path.resolve() for path in result.fratup_source_paths)
    if any(path.resolve() in inputs for path in paths.values()):
        raise ValueError("An output would overwrite an input file; choose a different output directory.")
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(paths[name], index=False)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=tuple(COHORT_NAMES), default="BO")
    parser.add_argument("--redcap-csv", default=None, help="Defaults to REDCap/Bologna or REDCap/Ravenna according to --cohort.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--codebook", help="Optional replacement JSON codebook; default uses the supplied refactor notebook.")
    parser.add_argument("--reference-csv", help="Optional May reference, for comparison only.")
    parser.add_argument("--sensor-csv", help="Optional existing T0 subject-level sensor aggregation to merge.")
    add_fratup_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = write_redcap_exports(args.redcap_csv, output_dir=args.output_dir, codebook_path=args.codebook,
                                 reference_csv=args.reference_csv, sensor_csv=args.sensor_csv, fratup_dir=args.fratup_dir, cohort=args.cohort)
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")
    print("FRAT-up scores/inputs imported from local CSVs." if args.fratup_dir is not None else "FRAT-up import explicitly disabled.")
    print("Exact six-month summaries stay deferred.")
    print("Individual falls include verified (form status 2), date allocation, and source-event provenance.")
    print("Inspect redcap_mapping.csv and redcap_codebook_issues.csv for historical mapping assumptions.")


if __name__ == "__main__":
    main()
