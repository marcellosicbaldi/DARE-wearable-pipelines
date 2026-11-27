"""Gyroscope evidence for an opt-in, respiration-assisted nonwear override.

A positive window can remove existing accelerometer nonwear only. This spectral
screen is experimental; absence of evidence never establishes nonwear.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields
from numbers import Real
import numpy as np
import pandas as pd
from scipy.signal import periodogram

AXES = ("gx", "gy", "gz")


@dataclass(frozen=True)
class RespirationOptions:
    """Disabled by default; configuration shared by both cohort adapters."""

    enabled: bool = False
    window_seconds: float = 60.0
    overlap: float = 0.5
    save_window_features: bool = True
    min_nonwear_bout_seconds: float = 0.0
    settings: RespirationSettings = field(default_factory=lambda: RespirationSettings())

    @classmethod
    def from_mapping(cls, values=None):
        values = dict(values or {})
        option_names = {"enabled", "window_seconds", "overlap", "save_window_features",
                        "min_nonwear_bout_seconds"}
        setting_names = {item.name for item in fields(RespirationSettings)} - {"band_hz"}
        unknown = values.keys() - option_names - setting_names - {"band_brpm"}
        if unknown:
            raise ValueError(f"Unknown respiration options: {sorted(unknown)}")
        options = {name: values[name] for name in option_names if name in values}
        for name in ("enabled", "save_window_features"):
            if name in options and type(options[name]) is not bool:
                raise ValueError(f"{name} must be a boolean.")
        settings = {name: values[name] for name in setting_names if name in values}
        if "band_brpm" in values:
            settings["band_hz"] = tuple(float(rate) / 60 for rate in values["band_brpm"])
        result = cls(**options, settings=RespirationSettings(**settings))
        result.validate(100.0)
        return result

    def validate(self, fs):
        self.settings.validate(fs)
        _validate_minimum_duration(self.min_nonwear_bout_seconds)
        if (not np.isfinite(self.window_seconds) or self.window_seconds <= 0
                or not 0 <= self.overlap < 1):
            raise ValueError("Choose a positive duration and overlap in [0, 1).")

@dataclass(frozen=True)
class RespirationSettings:
    """Exploratory starting values, to tune against independently reviewed bouts."""

    band_hz: tuple[float, float] = (10 / 60, 30 / 60)
    reference_band_hz: tuple[float, float] = (0.05, 2.0)
    min_band_rms_dps: float = 0.01
    min_band_power_fraction: float = 0.50
    min_peak_concentration: float = 0.50
    min_peak_to_background_db: float = 6.0
    peak_half_width_hz: float = 0.05
    max_gyro_std_dps: float | None = None

    def validate(self, fs: float) -> None:
        lo, hi = self.band_hz
        ref_lo, ref_hi = self.reference_band_hz
        if not (0 < ref_lo <= lo < hi < ref_hi < fs / 2):
            raise ValueError("Respiration band must lie inside the reference band and below Nyquist.")
        if not all(np.isfinite(value) for value in asdict(self).values()
                   if isinstance(value, (int, float))):
            raise ValueError("Thresholds must be finite.")
        if self.min_band_rms_dps <= 0 or self.peak_half_width_hz <= 0:
            raise ValueError("Amplitude and peak-width thresholds must be positive.")
        for value in (self.min_band_power_fraction, self.min_peak_concentration):
            if not 0 <= value <= 1:
                raise ValueError("Power fractions must be between zero and one.")
        if self.max_gyro_std_dps is not None and self.max_gyro_std_dps <= 0:
            raise ValueError("Optional motion guard must be positive, or None.")



@dataclass
class RespirationResult:
    windows: pd.DataFrame
    frequencies_hz: np.ndarray
    psd: np.ndarray | None  # window × axis × frequency; only up to reference_band_hz[1]
    window_seconds: float
    overlap: float
    fs: float
    n_samples: int
    segment_bounds: np.ndarray
    settings: RespirationSettings



def _bounds(bouts: pd.DataFrame, n_samples: int) -> np.ndarray:
    values = bouts[["start_sample", "stop_sample"]].to_numpy()
    if not len(values):
        # A header-only CSV has object dtypes because there are no values to infer.
        return np.empty((0, 2), dtype=np.int64)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Bout sample bounds must be finite integers.")
    bounds = values.astype(np.int64)
    if len(bounds) and (np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] > n_samples)
                        or np.any(bounds[:, 0] >= bounds[:, 1])
                        or np.any(bounds[1:, 0] < bounds[:-1, 1])):
        raise ValueError("Bout bounds must be sorted, non-overlapping and inside the recording.")
    return bounds.reshape(-1, 2)



def _segment_bounds(times: pd.DatetimeIndex, fs: float) -> np.ndarray:
    if times.hasnans:
        raise ValueError("Gyro timestamps must be unique and increasing.")
    delta = np.diff(times.asi8)
    if np.any(delta <= 0):
        raise ValueError("Gyro timestamps must be unique and increasing.")
    breaks = np.flatnonzero((delta > 1.5e9 / fs) | (delta < 0.5e9 / fs)) + 1
    return np.column_stack((np.r_[0, breaks], np.r_[breaks, len(times)]))



def analyze_gyro(gyro: pd.DataFrame, fs: float, *, window_seconds=60.0,
                 overlap=0.5, settings=None, batch_size=128,
                 axis_columns=AXES, time_column=None, store_psd=True, segment_bounds=None) -> RespirationResult:
    """Hann periodograms on unfiltered, full-rate gyro, independently per axis.

    Uses full windows only, restarting after timestamp gaps. PSDs are batched
    to bound temporary memory; no N × window_size copy of the week is made.
    """
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate must be finite and positive.")
    settings = settings or RespirationSettings()
    settings.validate(fs)
    if not np.isfinite(window_seconds) or window_seconds <= 0 or not 0 <= overlap < 1:
        raise ValueError("Choose a positive duration and overlap in [0, 1).")
    width = int(round(window_seconds * fs))
    hop_float = width * (1 - overlap)
    hop = int(round(hop_float))
    if (width < 4 or hop < 1 or not np.isclose(width, window_seconds * fs)
            or not np.isclose(hop, hop_float) or batch_size < 1):
        raise ValueError("Window and hop must be whole sample counts; batch_size must be positive.")
    times = pd.DatetimeIndex(gyro.index if time_column is None else gyro[time_column]).as_unit("ns")
    segments = _segment_bounds(times, fs)
    if segment_bounds is not None:
        bounds = np.asarray(segment_bounds)
        if (bounds.ndim != 2 or bounds.shape[1] != 2 or not len(bounds)
                or not np.isfinite(bounds).all() or not np.equal(bounds, np.floor(bounds)).all()):
            raise ValueError("Gyro segment bounds must be finite integer pairs.")
        bounds = bounds.astype(np.int64)
        if (bounds[0, 0] != 0 or bounds[-1, 1] != len(times)
                or np.any(bounds[:, 1] <= bounds[:, 0]) or np.any(bounds[1:, 0] != bounds[:-1, 1])):
            raise ValueError("Gyro segments must partition the observed samples.")
        breaks = np.unique(np.r_[segments[1:, 0], bounds[1:, 0]])
        segments = np.column_stack((np.r_[0, breaks], np.r_[breaks, len(times)]))
    if len(axis_columns) != 3:
        raise ValueError("Exactly three gyro columns are required, in x/y/z order.")
    values = [gyro[column].to_numpy(copy=False) for column in axis_columns]
    frequencies = np.fft.rfftfreq(width, d=1 / fs)
    keep = frequencies <= settings.reference_band_hz[1]
    f = frequencies[keep]
    band = (f > settings.band_hz[0]) & (f < settings.band_hz[1])
    reference = (f >= settings.reference_band_hz[0]) & (f <= settings.reference_band_hz[1])
    if band.sum() < 3 or (reference & ~band).sum() < 3:
        raise ValueError("Window is too short to resolve the configured spectral bands.")
    band_indices = np.flatnonzero(band)
    tables, spectra = [], []
    tiny = np.finfo(float).tiny
    for segment, (first, stop) in enumerate(segments):
        if stop - first < width:
            continue
        # This is a view. Only one batch is materialized for detrending/FFT.
        sliding = [np.lib.stride_tricks.sliding_window_view(axis[first:stop], width)
                   for axis in values]
        offsets = np.arange(0, stop - first - width + 1, hop)
        for batch_start in range(0, len(offsets), batch_size):
            local = offsets[batch_start:batch_start + batch_size]
            block = np.stack([axis[local] for axis in sliding], axis=1)  # batch × axis × samples
            valid = np.isfinite(block).all(axis=-1)
            safe = np.where(valid[..., None], block, 0.0)
            _, power = periodogram(safe, fs=fs, window="hann", detrend="linear",
                                   scaling="density", axis=-1)
            power = power[..., keep]
            peak_index = band_indices[np.argmax(power[..., band], axis=-1)]
            peak_hz = f[peak_index]
            peak_power = np.take_along_axis(power, peak_index[..., None], axis=-1)[..., 0]
            near_peak = np.abs(f - peak_hz[..., None]) <= max(settings.peak_half_width_hz, 1 / window_seconds)
            band_power = power[..., band].sum(axis=-1)
            reference_power = power[..., reference].sum(axis=-1)
            band_rms = np.sqrt(band_power * fs / width)
            fraction = band_power / np.maximum(reference_power, tiny)
            concentration = (power * (near_peak & band)).sum(axis=-1) / np.maximum(band_power, tiny)
            background = np.nanmedian(np.where(reference & ~near_peak, power, np.nan), axis=-1)
            peak_db = 10 * (np.log10(np.maximum(peak_power, tiny)) - np.log10(np.maximum(background, tiny)))
            before = np.take_along_axis(power, (peak_index - 1)[..., None], axis=-1)[..., 0]
            after_index = np.minimum(peak_index + 1, len(f) - 1)
            after = np.take_along_axis(power, after_index[..., None], axis=-1)[..., 0]
            local_peak = (peak_index < len(f) - 1) & (peak_power > before) & (peak_power > after)
            std = safe.std(axis=-1)
            positive = (valid & local_peak & (band_rms >= settings.min_band_rms_dps)
                        & (fraction >= settings.min_band_power_fraction)
                        & (concentration >= settings.min_peak_concentration)
                        & (peak_db >= settings.min_peak_to_background_db))
            if settings.max_gyro_std_dps is not None:
                positive &= std <= settings.max_gyro_std_dps
            starts = first + local
            stops = starts + width
            table = pd.DataFrame({
                "segment": segment, "start_sample": starts, "stop_sample": stops,
                "start": times[starts],
                "end": times[stops - 1] + pd.Timedelta(seconds=1 / fs),
                "center": times[starts + width // 2],
                "valid_axes": valid.sum(axis=1), "positive_axes": positive.sum(axis=1),
                "respiration_present": positive.any(axis=1),
            })
            for axis, name in enumerate(AXES):
                table[f"{name}_valid"] = valid[:, axis]
                table[f"{name}_positive"] = positive[:, axis]
                for feature, array in (
                    ("peak_hz", peak_hz), ("band_rms_dps", band_rms),
                    ("band_power_fraction", fraction), ("peak_concentration", concentration),
                    ("peak_to_background_db", peak_db), ("std_dps", std),
                ):
                    table[f"{name}_{feature}"] = np.where(valid[:, axis], array[:, axis], np.nan)
            power[~valid] = np.nan
            tables.append(table)
            if store_psd:
                spectra.append(power.astype(np.float32))
    if tables:
        windows = pd.concat(tables, ignore_index=True)
    else:
        windows = pd.DataFrame({
            **{name: pd.Series(dtype="int64") for name in
               ("segment", "start_sample", "stop_sample", "valid_axes", "positive_axes")},
            **{name: pd.Series(dtype=times.dtype) for name in ("start", "end", "center")},
            "respiration_present": pd.Series(dtype=bool),
        })
    psd = ((np.concatenate(spectra) if spectra else np.empty((0, 3, len(f)), dtype=np.float32))
           if store_psd else None)
    return RespirationResult(windows, f, psd, window_seconds, overlap,
                             fs, len(times), segments, settings)



def _merge(bounds: np.ndarray) -> np.ndarray:
    merged = []
    for start, stop in sorted(map(tuple, bounds)):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], int(stop))
        else:
            merged.append([int(start), int(stop)])
    return np.asarray(merged, dtype=np.int64).reshape(-1, 2)



def _covered_before(points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Sample-count integral of the union of half-open intervals."""
    bounds = _merge(bounds)
    if not len(bounds):
        return np.zeros(len(points), dtype=np.int64)
    prefix = np.r_[0, np.cumsum(bounds[:, 1] - bounds[:, 0])]
    count = np.searchsorted(bounds[:, 1], points, side="right")
    covered = prefix[count]
    partial = count < len(bounds)
    covered[partial] += np.maximum(points[partial] - bounds[count[partial], 0], 0)
    return covered



def nonwear_fraction(windows: pd.DataFrame, bouts: pd.DataFrame, n_samples: int) -> np.ndarray:
    bounds = _bounds(bouts, n_samples)
    start = windows.start_sample.to_numpy(dtype=np.int64)
    stop = windows.stop_sample.to_numpy(dtype=np.int64)
    return (_covered_before(stop, bounds) - _covered_before(start, bounds)) / (stop - start)



def _subtract(baseline: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    """Subtract a union of evidence intervals; preserve original bout boundaries."""
    evidence = _merge(evidence)
    remaining = []
    for start, stop in baseline:
        cursor = start
        first = np.searchsorted(evidence[:, 1], start, side="right")
        for left, right in evidence[first:]:
            if left >= stop:
                break
            if left > cursor:
                remaining.append((cursor, min(left, stop)))
            cursor = max(cursor, right)
            if cursor >= stop:
                break
        if cursor < stop:
            remaining.append((cursor, stop))
    return np.asarray(remaining, dtype=np.int64).reshape(-1, 2)



def _bout_table(bounds: np.ndarray, times: pd.DatetimeIndex, fs: float,
                breaks: np.ndarray) -> pd.DataFrame:
    # Evidence intervals may meet in sample order on opposite sides of a gap.
    # Split all exported bouts at those gaps, without assigning missing time.
    split = []
    for first, stop in bounds:
        inside = breaks[(breaks > first) & (breaks < stop)]
        edges = np.r_[first, inside, stop]
        split.extend(zip(edges[:-1], edges[1:]))
    intervals = np.asarray(split, dtype=np.int64).reshape(-1, 2)
    return pd.DataFrame({
        "start_sample": intervals[:, 0], "stop_sample": intervals[:, 1],
        "start": times[intervals[:, 0]],
        "end": times[intervals[:, 1] - 1] + pd.Timedelta(seconds=1 / fs),
        "duration_seconds": (intervals[:, 1] - intervals[:, 0]) / fs,
    })



def _validate_minimum_duration(seconds):
    if (isinstance(seconds, bool) or not isinstance(seconds, Real)
            or not np.isfinite(seconds) or seconds < 0):
        raise ValueError("min_nonwear_bout_seconds must be finite and non-negative.")


def filter_short_nonwear_bouts(bouts: pd.DataFrame, minimum_seconds: float):
    """Partition gap-split bouts by observed duration; retain the exact cutoff.

    Apply after respiration subtraction and timestamp-gap splitting. Never join
    bouts across wear or missing time. Zero reproduces the unfiltered output.
    The same operation can be used on saved half-open interval CSVs.
    """
    _validate_minimum_duration(minimum_seconds)
    durations = pd.to_numeric(bouts.duration_seconds, errors="raise")
    if not np.isfinite(durations).all() or (durations <= 0).any():
        raise ValueError("Nonwear bout durations must be finite and positive.")
    short = durations < minimum_seconds
    return (bouts.loc[~short].reset_index(drop=True).copy(),
            bouts.loc[short].reset_index(drop=True).copy())


def apply_override(result: RespirationResult, baseline: pd.DataFrame,
                   times: pd.DatetimeIndex, *, min_nonwear_bout_seconds=0.0) -> dict:
    """Any positive axis/window rescues only its overlap with existing nonwear.

    Overlapping positive windows are unioned before counting. Invalid windows,
    gaps and incomplete tails provide no evidence. Absence never creates nonwear.
    An optional final duration rule reassigns short residual bouts separately.
    """
    times = pd.DatetimeIndex(times).as_unit("ns")
    if len(times) != result.n_samples:
        raise ValueError("Spectral results and baseline must use the same sample sequence.")
    if not result.windows.start.equals(pd.Series(times[result.windows.start_sample.to_numpy()], name="start")):
        raise ValueError("Spectral window timestamps do not align with the supplied recording.")
    bounds = _bounds(baseline, len(times))
    windows = result.windows
    positive = windows.loc[windows.respiration_present, ["start_sample", "stop_sample"]].to_numpy()
    assessed = windows.loc[windows.valid_axes.gt(0), ["start_sample", "stop_sample"]].to_numpy()
    revised = _subtract(bounds, positive)
    rescued = _subtract(bounds, revised)
    counts = lambda spans: int((spans[:, 1] - spans[:, 0]).sum())
    base_count, revised_count, rescued_count = map(counts, (bounds, revised, rescued))
    covered = (_covered_before(bounds[:, 1], assessed) - _covered_before(bounds[:, 0], assessed)).sum()
    breaks = result.segment_bounds[1:, 0]
    before_duration_filter = _bout_table(revised, times, result.fs, breaks)
    final, short = filter_short_nonwear_bouts(before_duration_filter, min_nonwear_bout_seconds)
    short_count = int((short.stop_sample - short.start_sample).sum())
    return {
        "nonwear_bouts": final,
        "nonwear_bouts_before_duration_filter": before_duration_filter,
        "short_nonwear_bouts": short,
        "rescued_bouts": _bout_table(rescued, times, result.fs, breaks),
        "evidence_bouts": _bout_table(_merge(positive), times, result.fs, breaks),
        "summary": {
            "window_seconds": result.window_seconds, "overlap": result.overlap,
            "baseline_nonwear_hours": base_count / result.fs / 3600,
            "nonwear_before_duration_filter_hours": revised_count / result.fs / 3600,
            "revised_nonwear_hours": (revised_count - short_count) / result.fs / 3600,
            "respiration_reassigned_to_wear_hours": rescued_count / result.fs / 3600,
            "short_nonwear_reassigned_to_wear_hours": short_count / result.fs / 3600,
            "reassigned_to_wear_hours": (rescued_count + short_count) / result.fs / 3600,
            "min_nonwear_bout_seconds": float(min_nonwear_bout_seconds),
            "short_nonwear_bouts_reassigned": len(short),
            "baseline_nonwear_assessed_hours": covered / result.fs / 3600,
            "baseline_nonwear_unassessed_hours": (base_count - covered) / result.fs / 3600,
            "reassigned_percent_of_baseline_nonwear": 100 * (rescued_count + short_count) / base_count if base_count else 0.0,
        },
    }


def mask_bounds(mask):
    """Half-open sample intervals for a boolean state."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return edges.reshape(-1, 2)


def override_frame(frame, fs, options):
    """Preserve the exact production mask and add gyro evidence in place.

    Gyro is in degrees/s. Returned audit tables use half-open sample bounds and
    timestamps. Preprocessing converts final bouts to its legacy inclusive end.
    """
    options.validate(fs)
    result = analyze_gyro(frame, fs, window_seconds=options.window_seconds,
                          overlap=options.overlap, settings=options.settings,
                          axis_columns=("gyr_x", "gyr_y", "gyr_z"),
                          time_column="datetime", store_psd=False,
                          segment_bounds=frame.attrs.get("processing_segments"))
    times = pd.DatetimeIndex(frame.datetime).as_unit("ns")
    baseline = _bout_table(mask_bounds(~frame.wear.to_numpy(dtype=bool)), times,
                           fs, result.segment_bounds[1:, 0])
    changed = apply_override(result, baseline, times,
                             min_nonwear_bout_seconds=options.min_nonwear_bout_seconds)
    rescued = np.zeros(len(frame), dtype=bool)
    for first, stop in changed["rescued_bouts"][["start_sample", "stop_sample"]].to_numpy():
        rescued[first:stop] = True
    short = np.zeros(len(frame), dtype=bool)
    for first, stop in changed["short_nonwear_bouts"][["start_sample", "stop_sample"]].to_numpy():
        short[first:stop] = True
    frame["wear_accelerometer"] = frame.wear.to_numpy(copy=True)
    frame["wear_respiration_override"] = rescued
    frame["wear_short_nonwear_override"] = short
    frame["wear"] = frame.wear.to_numpy() | rescued | short
    result.windows["baseline_nonwear_fraction"] = nonwear_fraction(result.windows, baseline, len(frame))
    return {"result": result, "baseline_bouts": baseline, **changed}

