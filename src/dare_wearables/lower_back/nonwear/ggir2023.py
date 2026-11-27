"""Python port of GGIR's pinned 2023 matched-threshold nonwear profile.

Adapted from detect_nonwear_clipping.R and g.weardec.R in GGIR 3.3-9,
commit 5acd50f9283e02fe0efc98057978dc8e9d5629ab (Apache-2.0).
https://github.com/cran/GGIR/tree/5acd50f9283e02fe0efc98057978dc8e9d5629ab
See licenses/GGIR-LICENSE.txt and THIRD_PARTY_NOTICES.md for attribution.

Changes: translated to NumPy, fixed matched SD/range thresholds and >=2 axes,
and mapped decisions to exact input timestamps. No R runtime or downloads.
Only this profile is ported: no night filter, clipping excluded from nonwear.
Production uses a 60-minute window and disables recording-end assumptions.
The historical upstream 60-minute/end-corrected profile remains explicit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

GGIR_VERSION = "3.3-9"
GGIR_COMMIT = "5acd50f9283e02fe0efc98057978dc8e9d5629ab"
GGIR_SOURCES = {
    "load_params.R": "389d9db65d523ea10e40326a5422e636c8fb0ff30484e11666772de33a1bafc1",
    "detect_nonwear_clipping.R": "6fd2e93cf3b30779970d1c3d79858396ac5d0248ea004add52032cdb0bf857d4",
    "g.weardec.R": "08c0f07ccc76f4ef881038c95eb7476c54a181a8192c6e87b666f5b55e95bdee",
}
METHOD = "ggir_2023_matched"


@dataclass(frozen=True)
class NonwearOptions:
    """Production baseline; `current` explicitly selects the legacy detector."""

    method: str = METHOD
    window_seconds: float = 3600.0
    end_of_recording_correction: bool = False
    step_seconds: float = 900.0
    axes_required: int = 2
    postprocessing: bool = True

    def validate(self):
        if self.method not in (METHOD, "current"):
            raise ValueError("Choose nonwear.method = 'ggir_2023_matched' or 'current'.")
        if self.window_seconds not in (1800., 3600.):
            raise ValueError("nonwear.window_seconds must be 3600 or 1800.")
        if self.step_seconds not in (60., 900.):
            raise ValueError("nonwear.step_seconds must be 60 or 900.")
        if type(self.axes_required) is not int or self.axes_required not in (2, 3):
            raise ValueError("nonwear.axes_required must be 2 or 3.")
        if type(self.postprocessing) is not bool:
            raise ValueError("nonwear.postprocessing must be a boolean.")
        if self.method == "current" and (self.step_seconds != 900 or self.axes_required != 2 or not self.postprocessing):
            raise ValueError("New grid/axis/postprocessing options require ggir_2023_matched.")
        if type(self.end_of_recording_correction) is not bool:
            raise ValueError("nonwear.end_of_recording_correction must be a boolean.")

    @classmethod
    def from_mapping(cls, values=None):
        values = dict(values or {})
        # Accept obsolete R-adapter settings so existing TOML files still load.
        # They have no effect on the Python implementation or its provenance.
        values.pop("rscript", None)
        values.pop("source_cache", None)
        unknown = values.keys() - {"method", "window_seconds", "end_of_recording_correction",
                                  "step_seconds", "axes_required", "postprocessing"}
        if unknown:
            raise ValueError(f"Unknown nonwear options: {sorted(unknown)}")
        result = cls(**values)
        result.validate()
        return result


def prepare_runtime(options):
    """Validate the profile; production needs only the Python dependencies."""
    options.validate()


def detector_provenance(options):
    options.validate()
    if options.method == "current":
        return {"method": "current", "sd_thresh_mg": 13., "range_thresh_mg": 50., "axes_required": 2}
    return {"method": METHOD, "implementation": "python", "port_revision": 3,
            "ggir_version": GGIR_VERSION, "ggir_commit": GGIR_COMMIT,
            "source_sha256": dict(GGIR_SOURCES), "nonwear_approach": "2023",
            "sd_thresh_mg": 13., "range_thresh_mg": 50., "axes_required": options.axes_required,
            "window_seconds": float(options.window_seconds), "step_seconds": float(options.step_seconds), "clipping_in_nonwear_mask": False,
            "postprocessing": options.postprocessing,
            "start_edge_correction": options.postprocessing, "end_of_recording_correction": options.end_of_recording_correction,
            "reference_equivalent": (options.window_seconds == 3600 and options.step_seconds == 900
                                     and options.axes_required == 2 and options.postprocessing and options.end_of_recording_correction),
            "night_filter_max_hours": None, "statistics_target_hz": 5.}


def _sample_sd(values):
    """Sample SD with R's ordered reductions and platform long-double precision.

    NumPy's pairwise sum can cross the strict 13 mg cutoff for boundary inputs.
    cumsum preserves input order; the corrected mean and variance follow R's
    arithmetic, rounding variance to double before the final square root.
    """
    values = values.astype(np.longdouble)
    mean = np.cumsum(values)[-1] / len(values)
    mean += np.cumsum(values - mean)[-1] / len(values)
    deviation = values - np.longdouble(float(mean))
    variance = float(np.cumsum(deviation * deviation)[-1] / (len(values) - 1))
    return np.sqrt(variance)


def _epoch_scores(axes, freq, medium, count, *, window_epochs=4):
    """Port detect_nonwear_clipping's 2023 branch (finite calibrated g input)."""
    flags = np.zeros((count, 3), dtype=bool)
    scores = np.zeros(count, dtype=np.int64)
    clipping = np.zeros(count, dtype=float)
    stride = int(np.ceil(freq / 5))
    for epoch in range(count):
        first = epoch * medium
        for axis, values in enumerate(axes):
            # GGIR uses the full-rate epoch for clipping, not its 5 Hz subset.
            absolute = np.abs(values[first:first + medium])
            fraction = (1. if np.any(absolute > 11.25) else
                        np.count_nonzero(absolute > 7.5) / medium)
            clipping[epoch] = max(clipping[epoch], fraction)
            # Forward windows can include the incomplete final epoch.
            selected = values[first:first + window_epochs * medium:stride]
            if np.ptp(selected) < .050 and _sample_sd(selected) < .013:
                flags[epoch:min(epoch + window_epochs, count), axis] = True
        # Earlier windows may have marked different axes in the current epoch.
        scores[epoch] = flags[epoch].sum()
    # Match the upstream sequential promotion of isolated one-axis epochs.
    for epoch in np.flatnonzero(scores == 1):
        if 0 < epoch < count - 1 and scores[epoch - 1] > 1 and scores[epoch + 1] > 1:
            scores[epoch] = 2
    return scores, clipping


def _additional_nonwear(signal, added, count, *, end_of_recording_correction=True, step_seconds=900.):
    """GGIR gap/edge rules, retaining their physical durations at either grid."""
    three_h, six_h, one_h, day = (hours * 3600 / step_seconds for hours in (3, 6, 1, 24))
    starts = np.flatnonzero(np.diff(signal.astype(np.int8)) == 1) + 1
    stops = np.flatnonzero(np.diff(signal.astype(np.int8)) == -1) + 1
    for i in range(len(starts) - 1):
        gap = starts[i + 1] - stops[i]
        before = stops[i] - starts[i]
        after = stops[i + 1] - starts[i + 1]
        ratio = gap / (before + after)
        if ((gap < six_h and ratio < .3) or (gap < three_h and ratio < .8)
                # R uses 1-based padded indices in this test.
                or (end_of_recording_correction and starts[i] + 1 > count - day and gap < three_h and before > one_h)):
            added[stops[i]:starts[i + 1]] = True
    if len(starts):
        if starts[0] + 1 < three_h and starts[0] + 1 > 1:
            added[:starts[0]] = True
        if (end_of_recording_correction and stops[-1] + 1 > len(added) - three_h
                and stops[-1] + 1 != len(added)):
            added[stops[-1]:] = True
    # Upstream tests the enclosing nonWearEdgeCorrection variable, ignoring
    # its EdgeCorrection argument. Thus edges are corrected in all 3 passes.
    return added


def _wear_decisions(scores, clipping, *, end_of_recording_correction=True,
                    step_seconds=900., axes_required=2, postprocessing=True):
    """Port g.weardec with >=2 axes, edge correction, and no night filter."""
    initial = np.asarray(scores) >= axes_required
    clipped = np.asarray(clipping) > .3
    added = np.zeros(len(initial), dtype=bool)
    for _ in range(3 if postprocessing else 0):
        signal = np.r_[False, initial | added, False]
        added = _additional_nonwear(signal, np.r_[False, added, False], len(initial),
                                    end_of_recording_correction=end_of_recording_correction,
                                    step_seconds=step_seconds)[1:-1]
    return initial, added, initial | added, clipped


@dataclass
class GGIRNonwearResult:
    nonwear_df: pl.DataFrame
    window_df: pl.DataFrame
    provenance: dict


def detect_nonwear(acc_df, *, freq=100., options=None, segment_bounds=None):
    """Apply the GGIR 2023 Python port to calibrated x/y/z in g.

    Decisions cover complete, half-open configured-step sample intervals. Partial
    final epochs stay outside the decisions, matching the upstream comparison.
    Public bout timestamp ends are inclusive, preserving the pipeline contract.
    """
    options = options or NonwearOptions()
    options.validate()
    if options.method != METHOD:
        raise ValueError("detect_nonwear requires the ggir_2023_matched profile.")
    if not np.isfinite(freq) or freq <= 0 or not float(options.step_seconds * freq).is_integer():
        raise ValueError("GGIR requires a positive rate with whole-sample decision epochs.")
    times = pd.DatetimeIndex(acc_df["ts"].to_numpy()).as_unit("ns")
    if times.hasnans or np.any(np.diff(times.asi8) <= 0):
        raise ValueError("GGIR input timestamps must be strictly increasing.")
    axes = [acc_df[name].to_numpy() for name in ("x", "y", "z")]
    if any(not np.isfinite(values).all() for values in axes):
        raise ValueError("GGIR calibrated acceleration must be finite.")
    medium = int(options.step_seconds * freq)
    if segment_bounds is not None:
        bounds = np.asarray(segment_bounds)
        if (bounds.ndim != 2 or bounds.shape[1] != 2 or not len(bounds)
                or not np.isfinite(bounds).all() or not np.equal(bounds, np.floor(bounds)).all()):
            raise ValueError("GGIR segment bounds must be finite integer pairs.")
        bounds = bounds.astype(np.int64)
        if (bounds[0, 0] != 0 or bounds[-1, 1] != len(times)
                or np.any(bounds[:, 1] <= bounds[:, 0]) or np.any(bounds[1:, 0] != bounds[:-1, 1])):
            raise ValueError("GGIR segments must partition the observed samples.")
        tables, bouts, tails = [], [], []
        for segment, (first, stop) in enumerate(bounds):
            part = detect_nonwear(acc_df.slice(int(first), int(stop-first)), freq=freq, options=options)
            table = part.window_df.to_pandas()
            table[["start_sample", "stop_sample"]] += first
            table["segment"] = segment
            tables.append(table)
            bouts.append(part.nonwear_df.to_pandas())
            tail = int((stop-first) % medium)
            if tail:
                tails.append({"segment": segment, "start_sample": int(stop-tail), "stop_sample": int(stop)})
        provenance = detector_provenance(options) | {
            "segmented_for_timestamp_quality": True, "segments": bounds.tolist(),
            "unclassified_tail_samples": sum(row["stop_sample"]-row["start_sample"] for row in tails),
            "unclassified_tail_intervals": tails, "statistics_hz": freq/np.ceil(freq/5)}
        return GGIRNonwearResult(pl.from_pandas(pd.concat(bouts, ignore_index=True)),
                                 pl.from_pandas(pd.concat(tables, ignore_index=True)), provenance)
    count = len(times) // medium
    starts = np.arange(count, dtype=np.int64) * medium
    scorer = _epoch_scores
    if options.step_seconds == 60:
        from .minute_grid import minute_scores
        scorer = minute_scores
    scores, clipping = scorer(axes, freq, medium, count,
                              window_epochs=int(options.window_seconds / options.step_seconds))
    initial, added, final, clipped = _wear_decisions(scores, clipping,
        end_of_recording_correction=options.end_of_recording_correction,
        step_seconds=options.step_seconds, axes_required=options.axes_required,
        postprocessing=options.postprocessing)
    windows = pd.DataFrame({"start_sample": starts, "stop_sample": starts + medium,
                            "nonwear_score": scores, "axes_required": np.full(count, options.axes_required, dtype=np.int64),
                            "is_nonwear_window": initial, "is_added_nonwear": added,
                            "is_final_nonwear": final, "is_clipping": clipped,
                            "win_start": times[starts], "win_end": times[starts + medium - 1]})
    first = np.flatnonzero(final & np.r_[True, ~final[:-1]]) if count else np.empty(0, dtype=int)
    last = np.flatnonzero(final & np.r_[~final[1:], True]) if count else np.empty(0, dtype=int)
    bouts = pd.DataFrame({"start": times[starts[first]], "end": times[starts[last] + medium - 1]})
    provenance = detector_provenance(options)
    if not count:
        provenance["decision"] = f"no complete {options.step_seconds:g}-second epoch"
    provenance["unclassified_tail_samples"] = len(times) % medium
    provenance["statistics_hz"] = freq / np.ceil(freq / 5)
    return GGIRNonwearResult(pl.from_pandas(bouts), pl.from_pandas(windows), provenance)
