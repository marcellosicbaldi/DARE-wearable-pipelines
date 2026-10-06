"""Stable subject/visit joins for sensor-domain exports."""
import pandas as pd


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "subject" in out.columns:
        out["subject"] = out["subject"].astype(str)
    if "visit" in out.columns:
        out["visit"] = out["visit"].astype(str)
    return out


def _outer_merge_domain_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    normalized_frames = [
        _normalize_keys(frame)
        for frame in frames
        if frame is not None and not frame.empty
    ]
    if not normalized_frames:
        return pd.DataFrame(columns=["subject", "visit"])

    merged = normalized_frames[0]
    for frame in normalized_frames[1:]:
        merged = merged.merge(frame, on=["subject", "visit"], how="outer")
    return merged.sort_values(["subject", "visit"]).reset_index(drop=True)
