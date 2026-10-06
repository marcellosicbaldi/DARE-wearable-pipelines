from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dare_wearables.aggregation.merge import _normalize_keys, _outer_merge_domain_frames

from .activity_intensity import aggregate_activity_intensity
from .gait import aggregate_gait
from .heart_rate import DEFAULT_RECRUITMENT_TRACKER, aggregate_heart_rate
from .hrv import aggregate_discarded_percent, aggregate_hrv_windows, load_and_filter_hrv
from .sleep import AGGREGATION_METHODS, DEFAULT_SILVER_ROOT, DEFAULT_VISIT, aggregate_sleep
from .redcap import aggregate_redcap, merge_redcap_with_sensors
from fallspredict_gp_pipeline.redcap.fratup import DEFAULT_FRATUP_DIR, add_fratup_arguments


SLEEP_METHOD_CHOICES = (*AGGREGATION_METHODS, "both")






def _build_non_sleep_domain_frames(
    silver_root: str | Path,
    *,
    visit: str,
    recruitment_tracker_path: str | Path | None,
) -> list[pd.DataFrame]:
    _, hrv_filtered_df, _, hrv_discarded_df = load_and_filter_hrv(silver_root, visit=visit)
    hrv_df = aggregate_hrv_windows(hrv_filtered_df, visit=visit)
    hrv_discarded_wide_df, _ = aggregate_discarded_percent(hrv_discarded_df, visit=visit)
    heart_rate_df, _ = aggregate_heart_rate(
        silver_root,
        recruitment_tracker_path=recruitment_tracker_path,
        visit=visit,
    )
    gait_df, _ = aggregate_gait(silver_root, visit=visit)
    activity_df = aggregate_activity_intensity(silver_root, visit=visit)

    return [
        hrv_df,
        hrv_discarded_wide_df,
        heart_rate_df,
        gait_df,
        activity_df,
    ]


def aggregate_all(
    silver_root: str | Path = DEFAULT_SILVER_ROOT,
    *,
    visit: str = DEFAULT_VISIT,
    sleep_method: str = "median",
    recruitment_tracker_path: str | Path | None = DEFAULT_RECRUITMENT_TRACKER,
    redcap_csv: str | Path | None = None,
    redcap_codebook: str | Path | None = None,
    fratup_dir: str | Path | None = DEFAULT_FRATUP_DIR,
) -> pd.DataFrame:
    if sleep_method not in AGGREGATION_METHODS:
        raise ValueError(f"sleep_method must be one of {AGGREGATION_METHODS}; got {sleep_method!r}")

    clinical_df = aggregate_redcap(redcap_csv, visit=visit, codebook_path=redcap_codebook, fratup_dir=fratup_dir) if redcap_csv is not None else None
    sleep_df = aggregate_sleep(silver_root, visit=visit, method=sleep_method)  # type: ignore[arg-type]
    domain_frames = _build_non_sleep_domain_frames(
        silver_root,
        visit=visit,
        recruitment_tracker_path=recruitment_tracker_path,
    )
    sensors = _outer_merge_domain_frames([sleep_df, *domain_frames])
    return merge_redcap_with_sensors(clinical_df, sensors, visit=visit) if clinical_df is not None else sensors


def write_all_exports(
    silver_root: str | Path = DEFAULT_SILVER_ROOT,
    *,
    visit: str = DEFAULT_VISIT,
    output_dir: str | Path | None = None,
    sleep_method: str = "both",
    recruitment_tracker_path: str | Path | None = DEFAULT_RECRUITMENT_TRACKER,
    redcap_csv: str | Path | None = None,
    redcap_codebook: str | Path | None = None,
    fratup_dir: str | Path | None = DEFAULT_FRATUP_DIR,
) -> dict[str, tuple[Path, pd.DataFrame]]:
    if sleep_method not in SLEEP_METHOD_CHOICES:
        raise ValueError(f"sleep_method must be one of {SLEEP_METHOD_CHOICES}; got {sleep_method!r}")

    silver_root = Path(silver_root).expanduser()
    output_dir = Path(output_dir).expanduser() if output_dir is not None else silver_root / "aggregation"
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = AGGREGATION_METHODS if sleep_method == "both" else (sleep_method,)
    clinical_df = aggregate_redcap(redcap_csv, visit=visit, codebook_path=redcap_codebook, fratup_dir=fratup_dir) if redcap_csv is not None else None
    non_sleep_frames = _build_non_sleep_domain_frames(
        silver_root,
        visit=visit,
        recruitment_tracker_path=recruitment_tracker_path,
    )

    outputs: dict[str, tuple[Path, pd.DataFrame]] = {}
    for method in methods:
        sleep_df = aggregate_sleep(silver_root, visit=visit, method=method)  # type: ignore[arg-type]
        overall_df = _outer_merge_domain_frames([sleep_df, *non_sleep_frames])
        if clinical_df is not None:
            overall_df = merge_redcap_with_sensors(clinical_df, overall_df, visit=visit)
        output_path = output_dir / f"overall_{visit}_sleep_{method}.csv"
        overall_df.to_csv(output_path, index=False)
        outputs[str(method)] = (output_path, overall_df)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate all subject-level domain outputs together.")
    parser.add_argument(
        "--silver-root",
        default=str(DEFAULT_SILVER_ROOT),
        help="Root folder containing silver-layer outputs.",
    )
    parser.add_argument(
        "--visit",
        default=DEFAULT_VISIT,
        help="Visit to aggregate. Defaults to T0.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder for aggregate CSV exports. Defaults to <silver-root>/aggregation.",
    )
    parser.add_argument(
        "--sleep-method",
        choices=SLEEP_METHOD_CHOICES,
        default="both",
        help="Sleep aggregation to use in the overall export. Defaults to both.",
    )
    parser.add_argument(
        "--recruitment-tracker",
        default=str(DEFAULT_RECRUITMENT_TRACKER),
        help="Recruitment tracker Excel file used by the BeliefPPG HR aggregation.",
    )
    parser.add_argument("--redcap-csv", default=None, help="Optional REDCap export to add T0 clinical variables and all monthly follow-up observations.")
    parser.add_argument("--redcap-codebook", default=None, help="Optional verified disease/ATC JSON mapping for the REDCap export.")
    add_fratup_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = write_all_exports(
        args.silver_root,
        visit=args.visit,
        output_dir=args.output_dir,
        sleep_method=args.sleep_method,
        recruitment_tracker_path=args.recruitment_tracker,
        redcap_csv=args.redcap_csv,
        redcap_codebook=args.redcap_codebook,
        fratup_dir=args.fratup_dir,
    )
    for method, (path, df) in outputs.items():
        print(f"Wrote overall aggregation ({method} sleep): {path} ({len(df)} subjects, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
