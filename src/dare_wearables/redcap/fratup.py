"""Import locally computed FRAT-up CSVs; no scoring algorithm is included."""

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from dare_wearables.redcap.identity import normalize_subjects
from dare_wearables.redcap.cohorts import validate_cohort


DEFAULT_FRATUP_DIR = Path("~/dare-data/gold/fratup").expanduser()
IDENTIFIERS = {"record_id", "id", "patient_id", "subject"}
# Only equivalent concepts/encodings may reuse a clinical column. Binary risk
# factors are retained even when the clinical table contains their source score.
INPUT_ALIASES = {
    "sex": "sex", "age": "age", "livingalone": "living_alone",
    "hfallYN": "history_falls", "dizziness": "dizziness",
    "numberofmed": "daily_medications",
}
AUDIT_COLUMNS = ["source_file", "source_field", "target_column", "action",
                 "rows", "different_rows", "note"]


@dataclass
class FratupImport:
    clinical: pd.DataFrame
    inputs: pd.DataFrame
    mapping: list[dict]
    audit: pd.DataFrame
    source_paths: tuple[Path, ...]


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    return frame.apply(lambda col: col.str.strip().replace({"": pd.NA, "NA": pd.NA, "NaN": pd.NA, "nan": pd.NA}))


def _record_keys(values: pd.Series) -> pd.Series:
    # record_id is an opaque REDCap key, never a sensor participant ID.
    return values.astype("string").str.strip().str.replace(r"^(\d+)\.0+$", r"\1", regex=True)


def _subjects(frame: pd.DataFrame, identities: pd.DataFrame, label: str) -> pd.Series:
    if not IDENTIFIERS.intersection(frame.columns):
        raise ValueError(f"{label}: participant ID column required (record_id, id, patient_id, or subject); row-order matching is not supported.")
    lookup = identities.assign(record_id=_record_keys(identities["record_id"]))
    if lookup["record_id"].duplicated().any():
        raise ValueError("Ambiguous normalized REDCap record_id values.")
    by_record = lookup.set_index("record_id")["subject"]
    resolved = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in sorted(IDENTIFIERS.intersection(frame.columns)):
        present = frame[column].notna()
        values = frame.loc[present, column]
        ids = _record_keys(values).map(by_record) if column in {"record_id", "id"} else normalize_subjects(values)
        if ids.isna().any() or not ids.isin(lookup["subject"]).all():
            raise ValueError(f"{label}: {column} contains IDs outside the selected cohort's T0 baseline.")
        existing = resolved.loc[present]
        if (existing.notna() & existing.ne(ids)).any():
            raise ValueError(f"{label}: conflicting participant identifiers.")
        resolved.loc[present] = ids
    if resolved.isna().any() or resolved.duplicated().any():
        raise ValueError(f"{label}: missing or duplicate participant identifiers.")
    return resolved


def _numbers(values: pd.Series, label: str) -> pd.Series:
    try:
        result = pd.to_numeric(values, errors="raise").astype("Float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected numeric values or blanks/NA.") from exc
    if not np.isfinite(result.dropna().to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: non-finite numeric values.")
    return result


def _equivalent(values: pd.Series, clinical: pd.Series, field: str) -> pd.Series:
    missing = values.isna()
    if field == "sex":
        values = values.map({0: "M", 1: "F"})
    return values.eq(clinical).fillna(False) | (missing & clinical.isna())


def attach_fratup(clinical: pd.DataFrame, identities: pd.DataFrame,
                  folder: str | Path | None = DEFAULT_FRATUP_DIR, *, cohort: str = "BO") -> FratupImport:
    """Attach selected-cohort T0 scores and inputs, preserving the clinical prefix.

    Both input and result files must identify participants explicitly. Each is
    resolved independently against REDCap, so their row orders may differ.
    Input and result participant sets must agree; row-order matching is never used.
    """
    cohort = validate_cohort(cohort)
    out = clinical.copy()
    if folder is None:
        return FratupImport(out, pd.DataFrame(columns=["subject", "source_row"]), [],
                            pd.DataFrame(columns=AUDIT_COLUMNS), ())
    folder = Path(folder).expanduser()
    input_path = folder / f"fratup_input_{cohort}_T0.csv"
    candidates = [p for p in (folder / f"result_{cohort}_T0.csv", folder / f"results_{cohort}_T0.csv") if p.is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one result_{cohort}_T0.csv or results_{cohort}_T0.csv in {folder}; found {len(candidates)}.")
    score_path = candidates[0]
    inputs, results = _read(input_path), _read(score_path)
    if results.empty or inputs.empty:
        raise ValueError("FRAT-up input and result files must not be empty.")
    if "fratup" not in results:
        raise ValueError(f"{score_path.name}: missing fratup score column.")
    for frame, path in ((inputs, input_path), (results, score_path)):
        for field, expected in (("group", cohort), ("visit", "T0")):
            if field in frame and not frame[field].eq(expected).fillna(False).all():
                raise ValueError(f"{path.name}: {field} must be {expected}.")
    subjects = _subjects(results, identities, score_path.name)
    scores = _numbers(results["fratup"], score_path.name)
    if not scores.dropna().between(0, 1).all():
        raise ValueError("FRAT-up scores must be probabilities in [0, 1]; percentages are not converted implicitly.")
    input_subjects = _subjects(inputs, identities, input_path.name)
    if set(input_subjects) != set(subjects):
        raise ValueError("FRAT-up input/result participant sets disagree.")
    fields = [c for c in inputs if c not in IDENTIFIERS | {"group", "visit"}]
    risk = pd.DataFrame({c: _numbers(inputs[c], f"{input_path.name}:{c}") for c in fields})
    risk.index = pd.Index(input_subjects, name="subject")
    aligned = out.set_index("subject").reindex(risk.index)
    comparisons = {c: _equivalent(risk[c], aligned[target], c)
                   for c, target in INPUT_ALIASES.items() if c in risk and target in aligned}
    out["fratup"] = out["subject"].map(pd.Series(scores.to_numpy(), index=subjects, dtype="Float64"))
    mapping = [{"variable": "fratup", "status": "external_csv", "source_fields": f"{score_path}:fratup",
                "rule": "Locally computed probability imported by validated participant IDs; no FRAT-up algorithm runs here.",
                "missing_source_columns": ""}]
    audit = [{"source_file": str(input_path), "source_field": "", "target_column": "subject",
              "action": "participant_ids", "rows": len(inputs), "different_rows": 0,
              "note": "Input and result row order may differ; both are linked through validated participant IDs."},
             {"source_file": str(score_path), "source_field": "fratup", "target_column": "fratup",
              "action": "score_import", "rows": len(results), "different_rows": pd.NA,
              "note": f"{int(out.fratup.notna().sum())}/{len(out)} clinical participants have a score; unmatched/missing scores stay missing."}]
    for field in fields:
        same = comparisons.get(field)
        target = INPUT_ALIASES.get(field)
        reuse = same is not None and same.all()
        if not reuse:
            target = f"fratup_input_{field}"
            if target in out:
                raise ValueError(f"FRAT-up input column would overwrite {target}.")
            out[target] = out["subject"].map(risk[field])
            mapping.append({"variable": target, "status": "external_input", "source_fields": f"{input_path}:{field}",
                            "rule": "External FRAT-up input retained as supplied; participant_ids. No recomputation or clinical backfill.",
                            "missing_source_columns": ""})
        audit.append({"source_file": str(input_path), "source_field": field, "target_column": target,
                      "action": "already_present" if reuse else "added_input", "rows": len(inputs),
                      "different_rows": int((~same).sum()) if same is not None else pd.NA,
                      "note": "Equivalent clinical field; sex 0/1 corresponds to M/F." if reuse else "Preserve the input encoding independently of clinical scores/categories."})
    preserved = inputs.copy()
    if "subject" in preserved:
        preserved = preserved.rename(columns={"subject": "source_subject"})
    if "source_row" in preserved:
        raise ValueError("FRAT-up input has reserved column source_row.")
    preserved.insert(0, "source_row", np.arange(1, len(inputs) + 1))
    preserved.insert(0, "subject", input_subjects)
    if out.fratup.isna().any():
        warnings.warn(f"FRAT-up: {int(out.fratup.isna().sum())} clinical participants have no imported score.", stacklevel=2)
    return FratupImport(out, preserved, mapping, pd.DataFrame(audit, columns=AUDIT_COLUMNS), (input_path, score_path))


def add_fratup_arguments(parser) -> None:
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--fratup-dir", default=str(DEFAULT_FRATUP_DIR), help="Local FRAT-up input/result CSV folder; selects the requested cohort's T0 files.")
    options.add_argument("--no-fratup", dest="fratup_dir", action="store_const", const=None,
                         help="Explicitly skip external FRAT-up import; leave fratup missing.")
