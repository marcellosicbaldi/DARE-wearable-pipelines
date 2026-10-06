"""Combine locally exported Bologna/Ravenna sensors with each cohort's REDCap.

This final stage reads CSV/XLSX subject-level exports. It never runs sensor
processing, accesses the Ravenna server, or executes FRAT-up's private R code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dare_wearables.redcap.cohorts import COHORT_NAMES, DATA_ROOT, default_redcap_csv, validate_cohort
from dare_wearables.redcap.fratup import DEFAULT_FRATUP_DIR, add_fratup_arguments
from dare_wearables.redcap.identity import normalize_subjects
from dare_wearables.aggregation.analysis_exclusions import apply_analysis_exclusions, resolve_analysis_exclusions
from dare_wearables.aggregation.redcap import RedcapResult, merge_redcap_with_sensors, process_redcap


DEFAULT_BO_SENSOR_DIR = DATA_ROOT / "silver" / "aggregation"
DEFAULT_RA_SENSOR_DIR = DEFAULT_BO_SENSOR_DIR / "Ravenna"
KEYS = ["group", "subject", "visit"]
SUBJECT_ID_COLUMNS = frozenset({"subject", "subject_id", "patient_id", "record_id", "id", "source_subject"})


@dataclass
class CombinedResult:
    data: pd.DataFrame
    cohorts: dict[str, pd.DataFrame]
    redcap: dict[str, RedcapResult]
    column_coverage: pd.DataFrame
    summary: pd.DataFrame
    sources: pd.DataFrame


def _concat_union(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Preserve even wholly empty columns without pandas' all-NA dtype inference."""
    columns = list(dict.fromkeys(column for frame in frames for column in frame))
    populated = [frame.loc[:, frame.notna().any(axis=0)] for frame in frames]
    return pd.concat(populated, ignore_index=True, sort=False).reindex(columns=columns)


def _find_export(folder: Path, stem: str, *, required: bool = True) -> Path | None:
    candidates = [folder / f"{stem}{suffix}" for suffix in (".csv", ".xlsx")]
    found = [p for p in candidates if p.is_file()]
    if len(found) > 1:
        raise ValueError(f"Ambiguous export {stem} in {folder}: keep exactly one CSV or XLSX.")
    if not found and required:
        raise FileNotFoundError(f"Missing {stem}.csv or {stem}.xlsx in {folder}.")
    return found[0] if found else None


def read_sensor_export(path: str | Path, *, cohort: str) -> pd.DataFrame:
    """Read a single T0 subject-level table with validated cohort/subject keys."""
    cohort = validate_cohort(cohort)
    path = Path(path).expanduser()
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={"subject": "string", "visit": "string", "group": "string"})
    elif path.suffix.lower() == ".xlsx":
        with pd.ExcelFile(path) as book:
            if len(book.sheet_names) != 1:
                raise ValueError(f"{path}: expected exactly one sheet containing the subject-level export.")
            frame = pd.read_excel(book, sheet_name=0, dtype={"subject": "string", "visit": "string", "group": "string"})
    else:
        raise ValueError(f"Unsupported sensor export: {path}; use CSV or XLSX.")
    if not {"subject", "visit"}.issubset(frame):
        raise ValueError(f"{path}: subject and visit columns are required.")
    frame["subject"] = normalize_subjects(frame["subject"])
    frame["visit"] = frame["visit"].astype("string").str.strip().str.upper()
    if not frame["visit"].eq("T0").fillna(False).all():
        raise ValueError(f"{path}: expected only T0 rows.")
    if "group" in frame and not frame["group"].eq(cohort).fillna(False).all():
        raise ValueError(f"{path}: group must be {cohort}.")
    frame["group"] = cohort
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path}: duplicate cohort/subject/visit keys.")
    return frame


def load_sensor_exports(folder: str | Path, *, cohort: str, sleep_method: str = "mean") -> tuple[pd.DataFrame, list[Path]]:
    """Prefer an overall export; otherwise merge sleep, gait and activity exports.

    Mean and median sleep are alternative datasets, never duplicate rows or
    competing columns in the same dataset. All non-key columns are retained.
    """
    validate_cohort(cohort)
    if sleep_method not in {"mean", "median"}:
        raise ValueError("sleep_method must be mean or median.")
    folder = Path(folder).expanduser()
    overall = _find_export(folder, f"overall_T0_sleep_{sleep_method}", required=False)
    paths = [overall] if overall is not None else [
        _find_export(folder, f"sleep_T0_{sleep_method}"),
        _find_export(folder, "gait_T0_mean"),
        _find_export(folder, "activity_intensity_T0"),
    ]
    frames = [read_sensor_export(path, cohort=cohort) for path in paths]
    combined = frames[0]
    for frame in frames[1:]:
        overlap = set(combined).intersection(frame).difference(KEYS)
        if overlap:
            raise ValueError(f"Overlapping sensor domain columns for {cohort}: {sorted(overlap)}")
        combined = combined.merge(frame, on=KEYS, how="outer", validate="one_to_one")
    return combined.sort_values(KEYS).reset_index(drop=True), paths


def combine_cohort_frames(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Use one ordered column union, leaving unavailable cohort values missing."""
    if set(frames) != set(COHORT_NAMES):
        raise ValueError("The final dataset requires both BO and RA frames.")
    columns = list(KEYS)
    for cohort in COHORT_NAMES:
        frame = frames[cohort]
        if not set(KEYS).issubset(frame) or not frame["group"].eq(cohort).fillna(False).all():
            raise ValueError(f"Missing keys or incorrect group in {cohort} frame.")
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any() or not frame.columns.is_unique:
            raise ValueError(f"Missing/duplicate keys or columns in {cohort} frame.")
        columns.extend(c for c in frame if c not in columns)
    aligned = {c: frames[c].reindex(columns=columns).sort_values(KEYS).reset_index(drop=True) for c in COHORT_NAMES}
    coverage = pd.DataFrame([
        {"variable": field, **{
            key: value for cohort in COHORT_NAMES for key, value in (
                (f"{cohort}_column_present", field in frames[cohort]),
                (f"{cohort}_nonmissing_rows", int(aligned[cohort][field].notna().sum())),
            )}}
        for field in columns
    ])
    combined = _concat_union(list(aligned.values()))
    return combined, aligned, coverage


def build_combined_dataset(
    *, bologna_sensor_dir: str | Path = DEFAULT_BO_SENSOR_DIR,
    ravenna_sensor_dir: str | Path = DEFAULT_RA_SENSOR_DIR,
    bologna_redcap_csv: str | Path | None = None,
    ravenna_redcap_csv: str | Path | None = None,
    fratup_dir: str | Path | None = DEFAULT_FRATUP_DIR,
    sleep_method: str = "mean", bologna_codebook: str | Path | None = None,
    ravenna_codebook: str | Path | None = None,
    exclusions_config: str | Path | None = None,
    no_manual_exclusions: bool = False,
) -> CombinedResult:
    """Build one row per cohort/participant at T0 from local exports only."""
    exclusions = resolve_analysis_exclusions(exclusions_config, no_manual_exclusions=no_manual_exclusions)
    if sleep_method not in {"mean", "median"}:
        raise ValueError("sleep_method must be mean or median.")
    sources = []
    redcap = {}
    sensors = {}
    for cohort, sensor_dir, clinical_path, book in (
        ("BO", bologna_sensor_dir, bologna_redcap_csv, bologna_codebook),
        ("RA", ravenna_sensor_dir, ravenna_redcap_csv, ravenna_codebook),
    ):
        clinical_path = default_redcap_csv(cohort) if clinical_path is None else Path(clinical_path).expanduser()
        result = process_redcap(clinical_path, cohort=cohort, fratup_dir=fratup_dir, codebook_path=book)
        redcap[cohort] = result
        sensors[cohort], paths = load_sensor_exports(sensor_dir, cohort=cohort, sleep_method=sleep_method)
        for role, path in [("redcap", clinical_path), *[("sensor", p) for p in paths],
                           *[("fratup", p) for p in result.fratup_source_paths],
                           *([("codebook", Path(book).expanduser())] if book is not None else [])]:
            sources.append({"group": cohort, "role": role, "source_file": str(Path(path).resolve())})

    frames, summaries = {}, []
    for cohort, result in redcap.items():
        clinical = result.clinical.merge(result.followup_wide, on="subject", validate="one_to_one")
        # Retain the same REDCap population without expanding raw event fields
        # into the analysis table. Original fields remain in the event audit.
        clinical = result.events[["subject"]].drop_duplicates().merge(
            clinical, on="subject", how="left", validate="one_to_one",
        )
        clinical["group"] = cohort
        clinical["has_redcap"] = True
        sensor = sensors[cohort].copy()
        if "has_redcap" in sensor or "has_sensors" in sensor:
            raise ValueError("Sensor export uses reserved presence-flag columns.")
        sensor["has_sensors"] = True
        frame = merge_redcap_with_sensors(clinical, sensor, cohort=cohort)
        for flag in ("has_redcap", "has_sensors"):
            frame[flag] = frame[flag].eq(True).astype(int)
        frame = apply_analysis_exclusions(frame, exclusions=exclusions)
        frames[cohort] = frame
        summaries.append({"group": cohort, "participants": len(frame),
                          "redcap_baseline_participants": len(result.clinical),
                          "sensor_participants": len(sensor),
                          "matched_participants": int((frame.has_redcap.eq(1) & frame.has_sensors.eq(1)).sum()),
                          "clinical_only": int((frame.has_redcap.eq(1) & frame.has_sensors.eq(0)).sum()),
                          "sensor_only": int((frame.has_redcap.eq(0) & frame.has_sensors.eq(1)).sum()),
                          "fratup_scores": int(frame.fratup.notna().sum()),
                          "rmssd_sdnn_exclusions": int(frame.rmssd_sdnn_exclusion.sum()),
                          "sleep_circadian_exclusions": int(frame.sleep_circadian_exclusion.sum())})
    data, aligned, coverage = combine_cohort_frames(frames)
    if exclusions_config is not None:
        sources.append({"group": "BO+RA", "role": "manual_exclusions",
                        "source_file": str(Path(exclusions_config).expanduser().resolve())})
    return CombinedResult(data, aligned, redcap, coverage, pd.DataFrame(summaries), pd.DataFrame(sources))


def write_combined_exports(
    *, output_dir: str | Path = DATA_ROOT / "gold" / "aggregation",
    save_subject_id: bool = True, **kwargs,
) -> dict[str, Path]:
    """Write CSVs, optionally omitting known participant ID columns at save time.

    IDs remain available during joins, exclusions and coverage calculations.
    This option removes columns, not identifiers embedded in arbitrary text.
    """
    if not isinstance(save_subject_id, bool):
        raise TypeError("save_subject_id must be a boolean (True or False).")
    result = build_combined_dataset(**kwargs)
    method = kwargs.get("sleep_method", "mean")
    suffix = "" if method == "mean" else f"_sleep_{method}"
    tables = {f"combined_T0{suffix}": result.data,
              f"bologna_T0{suffix}": result.cohorts["BO"],
              f"ravenna_T0{suffix}": result.cohorts["RA"],
              f"cohort_summary_T0{suffix}": result.summary,
              f"column_coverage_T0{suffix}": result.column_coverage,
              f"source_files_T0{suffix}": result.sources}
    for name in ("events", "falls", "fall_monthly", "followup_monthly", "mapping", "codebook_issues", "fratup_inputs", "fratup_import"):
        tables[f"redcap_{name}"] = _concat_union([getattr(r, name).assign(group=c) for c, r in result.redcap.items()])
    if not save_subject_id:
        tables = {
            name: table.drop(columns=[c for c in table if c.strip().lower() in SUBJECT_ID_COLUMNS])
            for name, table in tables.items()
        }
        coverage_name = f"column_coverage_T0{suffix}"
        coverage = tables[coverage_name]
        tables[coverage_name] = coverage.loc[
            ~coverage["variable"].str.strip().str.lower().isin(SUBJECT_ID_COLUMNS)
        ]
    folder = Path(output_dir).expanduser()
    paths = {name: folder / f"fallspredict_{name}.csv" for name in tables}
    inputs = {Path(p).resolve() for p in result.sources["source_file"]}
    if any(p.resolve() in inputs for p in paths.values()):
        raise ValueError("An output would overwrite an input; choose another output directory.")
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(paths[name], index=False, na_rep="NaN")
    return paths


def _parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value not in {"true", "false"}:
        raise argparse.ArgumentTypeError("Expected TRUE or FALSE (case-insensitive).")
    return value == "true"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bologna-sensor-dir", default=str(DEFAULT_BO_SENSOR_DIR))
    parser.add_argument("--ravenna-sensor-dir", default=str(DEFAULT_RA_SENSOR_DIR))
    parser.add_argument("--bologna-redcap-csv", default=None)
    parser.add_argument("--ravenna-redcap-csv", default=None)
    parser.add_argument("--bologna-codebook", default=None)
    parser.add_argument("--ravenna-codebook", default=None)
    parser.add_argument("--sleep-method", choices=("mean", "median"), default="mean")
    parser.add_argument("--save-subject-id", type=_parse_bool, default=True, metavar="TRUE|FALSE",
                        help="Save participant ID columns in all CSVs (default: TRUE). FALSE removes known ID columns after processing.")
    parser.add_argument("--output-dir", default=str(DATA_ROOT / "gold" / "aggregation"))
    manual = parser.add_mutually_exclusive_group(required=True)
    manual.add_argument("--exclusions-config", help="Private TOML containing the study's manual exclusion IDs.")
    manual.add_argument("--no-manual-exclusions", action="store_true",
                        help="Explicitly apply no participant-specific manual exclusions; coverage rules still apply.")
    add_fratup_arguments(parser)
    return parser


def main() -> None:
    paths = write_combined_exports(**vars(build_parser().parse_args()))
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")
    print("BO and RA exports share exactly the combined column schema; unavailable values are NaN.")
    print("Inspect cohort_summary, column_coverage and redcap_codebook_issues.")


if __name__ == "__main__":
    main()
