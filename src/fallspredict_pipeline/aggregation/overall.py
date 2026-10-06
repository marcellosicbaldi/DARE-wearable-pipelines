from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dare_wearables.aggregation.merge import _normalize_keys, _outer_merge_domain_frames

from .activity_intensity import aggregate_activity_intensity
from .gait import aggregate_gait
from .sleep import AGGREGATION_METHODS, DEFAULT_SILVER_ROOT, DEFAULT_VISIT, aggregate_sleep


SLEEP_METHOD_CHOICES = (*AGGREGATION_METHODS, "both")






def _build_non_sleep_domain_frames(
    silver_root: str | Path,
    *,
    visit: str,
) -> list[pd.DataFrame]:
    gait_df, _ = aggregate_gait(silver_root, visit=visit)
    activity_df = aggregate_activity_intensity(silver_root, visit=visit)
    return [
        gait_df,
        activity_df,
    ]


def aggregate_all(
    silver_root: str | Path = DEFAULT_SILVER_ROOT,
    *,
    visit: str = DEFAULT_VISIT,
    sleep_method: str = "median",
) -> pd.DataFrame:
    if sleep_method not in AGGREGATION_METHODS:
        raise ValueError(f"sleep_method must be one of {AGGREGATION_METHODS}; got {sleep_method!r}")

    sleep_df = aggregate_sleep(silver_root, visit=visit, method=sleep_method)  # type: ignore[arg-type]
    domain_frames = _build_non_sleep_domain_frames(silver_root, visit=visit)
    return _outer_merge_domain_frames([sleep_df, *domain_frames])


def write_all_exports(
    silver_root: str | Path = DEFAULT_SILVER_ROOT,
    *,
    visit: str = DEFAULT_VISIT,
    output_dir: str | Path | None = None,
    sleep_method: str = "both",
) -> dict[str, tuple[Path, pd.DataFrame]]:
    if sleep_method not in SLEEP_METHOD_CHOICES:
        raise ValueError(f"sleep_method must be one of {SLEEP_METHOD_CHOICES}; got {sleep_method!r}")

    silver_root = Path(silver_root).expanduser()
    output_dir = Path(output_dir).expanduser() if output_dir is not None else silver_root / "aggregation"
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = AGGREGATION_METHODS if sleep_method == "both" else (sleep_method,)
    non_sleep_frames = _build_non_sleep_domain_frames(silver_root, visit=visit)

    outputs: dict[str, tuple[Path, pd.DataFrame]] = {}
    for method in methods:
        sleep_df = aggregate_sleep(silver_root, visit=visit, method=method)  # type: ignore[arg-type]
        overall_df = _outer_merge_domain_frames([sleep_df, *non_sleep_frames])
        output_path = output_dir / f"overall_{visit}_sleep_{method}.csv"
        overall_df.to_csv(output_path, index=False)
        outputs[str(method)] = (output_path, overall_df)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate DARE-FALLSPREDICT subject-level domain outputs together.")
    parser.add_argument(
        "--silver-root",
        default=str(DEFAULT_SILVER_ROOT),
        help="Root folder containing DARE-FALLSPREDICT SILVER outputs.",
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = write_all_exports(
        args.silver_root,
        visit=args.visit,
        output_dir=args.output_dir,
        sleep_method=args.sleep_method,
    )
    for method, (path, df) in outputs.items():
        print(f"Wrote overall aggregation ({method} sleep): {path} ({len(df)} subjects, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
