"""Notebook helpers around the shared gyro spectral screen.

This is a tunable respiration-like signal screen, not a validated respiration
detector. It can only remove existing nonwear labels; it never creates nonwear.
"""
from __future__ import annotations

from datetime import time as clock_time
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dare_wearables.lower_back.nonwear.respiration import (
    AXES, RespirationSettings, RespirationResult, analyze_gyro, apply_override,
    filter_short_nonwear_bouts,
    nonwear_fraction, _bounds, _segment_bounds, _merge, _covered_before,
    _subtract, _bout_table,
)
METHODS = ("current", "ggir_2023_matched", "ggir_2023_matched_3axes")


def find_comparison_run(output_root, subject="900001", visit="T0", *,
                        method="ggir_2023_matched", run=None) -> Path:
    """Choose a completed comparison containing this recording and baseline.

    Old two-axis matched runs are usable: the retired 150 mg variant is ignored.
    No comparisons or R processes are started here.
    """
    if method not in METHODS:
        raise ValueError("Choose current or a matched-threshold GGIR method.")
    candidates = ([Path(run).expanduser().resolve()] if run is not None else
                  sorted(Path(output_root).expanduser().glob("run_*"), reverse=True))
    for candidate in candidates:
        manifest_path = candidate / "manifest.json"
        folder = candidate / str(subject) / str(visit)
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("status") == "completed" and
                (folder / "recording.json").is_file() and
                (folder / f"{method}_bouts.csv").is_file()):
            return candidate.resolve()
    raise FileNotFoundError(
        f"No completed comparison for {subject}/{visit} containing {method}. "
        "Set COMPARISON_RUN explicitly or first run the nonwear comparison notebook."
    )


def load_recording(run, subject="900001", visit="T0"):
    """Read gyro once using the comparison's exact input, sample cap and units."""
    from dare_wearables.lower_back.io.mcroberts_loader import read_dp7_as_dataframe_fast

    folder = Path(run) / str(subject) / str(visit)
    info = json.loads((folder / "recording.json").read_text())
    source = Path(info["input"]).expanduser()
    stat = source.stat()
    if (stat.st_size != info["input_size_bytes"] or
            stat.st_mtime_ns != info["input_mtime_ns"]):
        raise ValueError("The original OMX changed since this comparison; generate a new comparison.")
    raw = read_dp7_as_dataframe_fast(source, verbose=0)
    fs = float(raw.attrs.get("sample_rate_hz", np.nan))
    if not np.isfinite(fs) or not np.isclose(fs, info["sample_rate_hz"]):
        raise ValueError("Raw and comparison sampling rates disagree.")
    n = int(info["n_samples"])
    if n < 1 or len(raw) < n:
        raise ValueError("Raw recording is shorter than the comparison input.")
    # Keep only the analysed portion and gyro; avoid retaining full-week ACC.
    gyro = raw.iloc[:n][["datetime", *AXES]].set_index("datetime")
    del raw
    gyro.index = pd.DatetimeIndex(gyro.index).as_unit("ns")
    if (gyro.index[0] != pd.Timestamp(info["start"]) or
            gyro.index[-1] + pd.Timedelta(seconds=1 / fs) != pd.Timestamp(info["end"])):
        raise ValueError("Raw and comparison timestamps do not align.")
    baselines = {}
    for method in METHODS:
        file = folder / f"{method}_bouts.csv"
        if file.is_file():
            bouts = pd.read_csv(file, parse_dates=["start", "end"])
            _bounds(bouts, n)
            baselines[method] = bouts
    return gyro, fs, info, baselines


def _daytime_spans(left, right, masked_daytime):
    """Daily display masks in recording-local clock time, clipped to the view."""
    if masked_daytime is None:
        return []
    day_start, day_end = map(clock_time.fromisoformat, masked_daytime)
    if day_start.tzinfo is not None or day_end.tzinfo is not None or day_start >= day_end:
        raise ValueError("Choose daytime start < end as local HH:MM clock times.")
    spans = []
    for date in pd.date_range(left.normalize(), right.normalize(), freq="D"):
        first = pd.Timestamp.combine(date.date(), day_start).tz_localize(left.tz)
        stop = pd.Timestamp.combine(date.date(), day_end).tz_localize(left.tz)
        if first < right and stop > left:
            spans.append((max(left, first), min(right, stop)))
    return spans


def plot_overview(gyro, result, baseline, override, *, start=None, end=None,
                   masked_daytime=("09:00", "21:00"), brpm_limits=(10.0, 30.0),
                   brpm_ylim=(0.0, 50.0)):
    """Full-time spectra/bout labels, with nighttime BR dots and evidence.

    The daytime mask, spectral y-limits and strict marker-rate bounds affect drawing only.
    Detection, override masks and exported duration summaries are unchanged.
    ``gyro`` supplies the recording's time bounds; no raw trace is plotted.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if result.psd is None:
        raise ValueError("Plotting spectra requires analyze_gyro(..., store_psd=True).")
    left = pd.Timestamp(start) if start is not None else gyro.index[0]
    right = pd.Timestamp(end) if end is not None else gyro.index[-1] + pd.Timedelta(seconds=1 / result.fs)
    if left >= right:
        raise ValueError("Plot start must precede plot end.")
    if right <= gyro.index[0] or left > gyro.index[-1]:
        raise ValueError("Plot interval contains no samples.")
    lower, upper = brpm_limits
    if not np.isfinite([lower, upper]).all() or not 0 < lower < upper:
        raise ValueError("Choose finite positive BR limits with lower < upper.")
    ymin, ymax = brpm_ylim
    if not np.isfinite([ymin, ymax]).all() or not 0 <= ymin < ymax:
        raise ValueError("Choose finite spectral y-limits with 0 <= lower < upper.")
    frequencies_bpm = result.frequencies_hz * 60
    frequency_visible = (frequencies_bpm >= ymin) & (frequencies_bpm <= ymax)
    if not frequency_visible.any():
        raise ValueError("No spectral bins lie inside the requested y-limits.")
    selected = (result.windows.end > left) & (result.windows.start < right)
    ids = np.flatnonzero(selected)
    if not len(ids):
        raise ValueError("No analysed windows overlap this plot interval.")
    daytime = _daytime_spans(left, right, masked_daytime)
    # PSD remains visible at all hours; the time mask applies only to BR dots
    # and the respiration-like evidence row, not to spectra or nonwear bouts.
    db = 10 * np.log10(np.maximum(result.psd[ids][..., frequency_visible], 1e-20))
    finite = db[np.isfinite(db)]
    low, high = np.percentile(finite, [5, 99]) if len(finite) else (-80, 0)
    high = max(high, low + 1)
    fig, axes = plt.subplots(4, 1, figsize=(17, 10), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.2]})
    for axis, name in enumerate(AXES):
        ax = axes[axis]
        # Separate segment meshes leave missing recording time visibly blank.
        for _, group in result.windows.iloc[ids].groupby("segment", sort=False):
            indices = group.index.to_numpy()
            centers = mdates.date2num(pd.DatetimeIndex(group.center).to_pydatetime())
            hop_days = result.window_seconds * (1 - result.overlap) / 86400
            edges = np.r_[centers - hop_days / 2, centers[-1] + hop_days / 2]
            displayed_frequencies = frequencies_bpm[frequency_visible]
            half_bin = result.fs / round(result.window_seconds * result.fs) * 30
            freq_edges = np.r_[displayed_frequencies - half_bin, displayed_frequencies[-1] + half_bin]
            power_db = 10 * np.log10(np.maximum(result.psd[indices, axis, :][:, frequency_visible], 1e-20))
            night = np.ones(len(group), dtype=bool)
            for first, stop in daytime:
                night &= ~((group.center >= first) & (group.center < stop)).to_numpy()
            mesh = ax.pcolormesh(edges, freq_edges, power_db.T, cmap="magma", vmin=low, vmax=high,
                                 shading="flat", rasterized=True)
            peak_brpm = group[f"{name}_peak_hz"] * 60
            positive = group[f"{name}_positive"] & night & peak_brpm.gt(lower) & peak_brpm.lt(upper)
            ax.scatter(group.loc[positive, "center"], group.loc[positive, f"{name}_peak_hz"] * 60,
                       s=5, color="#49e3df")
        for hz in result.settings.band_hz:
            if ymin < hz * 60 < ymax:
                ax.axhline(hz * 60, color="white", ls="--", lw=.7)
        ax.set(ylabel=f"{name}\nBR (brpm)", ylim=(ymin, ymax))
    # One shared colorbar keeps spectral maps and label rows time-aligned.
    fig.colorbar(mesh, ax=list(axes), pad=.01, fraction=.025, label="PSD dB [(°/s)²/Hz]")
    rows = [(baseline, "Original nonwear", "#c0392b"),
            (override["evidence_bouts"], "Respiration-like evidence", "#ca8a19"),
            (override["rescued_bouts"], "Wear: respiration", "#19715b")]
    if override["summary"].get("min_nonwear_bout_seconds", 0) > 0:
        rows.append((override["short_nonwear_bouts"], "Wear: short nonwear bout", "#8e44ad"))
    rows.append((override["nonwear_bouts"], "Remaining nonwear", "#315fc4"))
    for row, (bouts, label, color) in enumerate(rows):
        visible = bouts.loc[(bouts.end > left) & (bouts.start < right)]
        for bout in visible.itertuples(index=False):
            pieces = [(max(left, bout.start), min(right, bout.end))]
            if label == "Respiration-like evidence":
                for first, stop in daytime:
                    retained = []
                    for piece_start, piece_end in pieces:
                        if piece_end <= first or piece_start >= stop:
                            retained.append((piece_start, piece_end))
                        else:
                            if piece_start < first:
                                retained.append((piece_start, first))
                            if piece_end > stop:
                                retained.append((stop, piece_end))
                    pieces = retained
            for piece_start, piece_end in pieces:
                axes[-1].plot([piece_start, piece_end], [row, row],
                              color=color, lw=6, solid_capstyle="butt")
    axes[-1].set_yticks(range(len(rows)), [row[1] for row in rows])
    axes[-1].set_ylim(-.5, len(rows) - .5)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    for ax in axes:
        ax.set_xlim(left, right)
    mask_label = (f"dots/evidence hidden {masked_daytime[0]}–{masked_daytime[1]}" if masked_daytime is not None
                  else "no daytime mask")
    fig.suptitle(f"{result.window_seconds:g} s windows · {result.overlap:.0%} overlap · spectral range {ymin:g}–{ymax:g} brpm\n"
                 f"cyan = positive axis with {lower:g} < BR < {upper:g} brpm · {mask_label}")
    return fig


def plot_window(gyro, result: RespirationResult, window_index: int):
    """All original gyro samples and the exact PSD used by the screen."""
    import matplotlib.pyplot as plt

    row = result.windows.loc[window_index]
    raw = gyro.iloc[int(row.start_sample):int(row.stop_sample)]
    seconds = (raw.index - raw.index[0]).total_seconds()
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex="col")
    for axis, name in enumerate(AXES):
        axes[axis, 0].plot(seconds, raw[name] - raw[name].mean(), lw=.8)
        axes[axis, 0].set(ylabel=f"{name} (°/s)\nmean removed")
        axes[axis, 1].semilogy(result.frequencies_hz * 60,
                               np.maximum(result.psd[window_index, axis], 1e-20))
        axes[axis, 1].axvspan(*np.asarray(result.settings.band_hz) * 60, color="#19715b", alpha=.15)
        axes[axis, 1].axvline(row[f"{name}_peak_hz"] * 60, color="#c0392b", ls="--")
        axes[axis, 1].set(ylabel="PSD [(°/s)²/Hz]",
                          title=f"{name}: positive={row[f'{name}_positive']} · band RMS={row[f'{name}_band_rms_dps']:.3g} °/s\n"
                                f"band fraction={row[f'{name}_band_power_fraction']:.2f} · concentration={row[f'{name}_peak_concentration']:.2f} · peak/background={row[f'{name}_peak_to_background_db']:.1f} dB")
    axes[0, 0].set_title(f"Unfiltered gyro · {result.fs:g} Hz")
    axes[-1, 0].set_xlabel("Seconds from window start")
    axes[-1, 1].set_xlabel("Frequency (cycles/min; candidate only, not validated breathing rate)")
    fig.suptitle(f"{row.start} — {row.end} · positive in {row.positive_axes} axis/axes")
    fig.tight_layout()
    return fig
