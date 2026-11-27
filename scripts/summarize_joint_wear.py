"""Reconstruct T0 pipeline-defined wear coverage from immutable saved outputs."""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

NS_HOUR = 3_600_000_000_000
PROTOCOL = 168 * NS_HOUR
METRICS = ("mcroberts", "empatica", "both")
LABELS = {"mcroberts": "McRoberts", "empatica": "Empatica", "both": "Both simultaneously"}


def union(intervals):
    result = []
    for start, end in sorted((int(s), int(e)) for s, e in intervals if e > s):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def subtract(intervals, exclusions):
    excluded, result = union(exclusions), []
    for start, end in union(intervals):
        cursor = start
        for left, right in excluded:
            if right <= cursor:
                continue
            if left >= end:
                break
            if left > cursor:
                result.append([cursor, min(left, end)])
            cursor = max(cursor, right)
            if cursor >= end:
                break
        if cursor < end:
            result.append([cursor, end])
    return result


def intersect(first, second):
    first, second = union(first), union(second)
    i = j = 0
    result = []
    while i < len(first) and j < len(second):
        start, end = max(first[i][0], second[j][0]), min(first[i][1], second[j][1])
        if start < end:
            result.append([start, end])
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return union(result)


def clipped(intervals, lo, hi):
    return intersect(intervals, [[lo, hi]])


def hours(intervals):
    return sum(e - s for s, e in union(intervals)) / NS_HOUR


def stamp(value):
    return pd.Timestamp(int(value)).isoformat(sep=" ")


def read_intervals(frame):
    intervals = []
    for start, end in zip(frame.start, frame.end):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        require(not pd.isna(start) and not pd.isna(end) and start.tz is None and end.tz is None
                and start < end, "Intervals require increasing, timezone-naive stored timestamps.")
        intervals.append([start.value, end.value])
    return intervals


def partition(mc, emp, lo, hi):
    """Independent sweep over interval endpoints; yields four exhaustive states."""
    events = {lo: [0, 0], hi: [0, 0]}
    for axis, intervals in enumerate((clipped(mc, lo, hi), clipped(emp, lo, hi))):
        for start, end in intervals:
            events.setdefault(start, [0, 0])[axis] += 1
            events.setdefault(end, [0, 0])[axis] -= 1
    points = sorted(events)
    state = [0, 0]
    result = []
    for left, right in zip(points, points[1:]):
        state = [state[i] + events[left][i] for i in range(2)]
        require(state[0] in (0, 1) and state[1] in (0, 1), 'Coverage partition contains overlapping or invalid intervals.')
        name = {(1, 1): "both", (1, 0): "mcroberts_only", (0, 1): "empatica_only", (0, 0): "neither"}[tuple(state)]
        if result and result[-1][0] == name and result[-1][2] == left:
            result[-1][2] = right
        else:
            result.append([name, left, right])
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze(empatica_dir, mcroberts_run, output, *, plots=True):
    empatica_dir, mcroberts_run, output = (Path(p).resolve() for p in (empatica_dir, mcroberts_run, output))
    if output.exists():
        raise FileExistsError("Choose a new report directory; existing outputs are preserved.")
    for source in (empatica_dir, mcroberts_run):
        require(source != output and source not in output.parents and output not in source.parents,
                "Report directory must be separate from both source trees.")
    sources = {}

    def load_bytes(path):
        payload = path.read_bytes()
        sources[path] = dict(path=str(path), size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        return payload

    def load_json(path):
        return json.loads(load_bytes(path))

    def load_csv(path, **kwargs):
        return pd.read_csv(io.BytesIO(load_bytes(path)), **kwargs)

    status = load_json(mcroberts_run / "run_status.json")
    require(status["status"] == "completed", "McRoberts run must be complete.")
    emp = load_csv(empatica_dir / "subject_summary.csv", dtype={"subject": str, "visit": str})
    emp_intervals = load_csv(empatica_dir / "intervals.csv", dtype={"subject": str, "visit": str})
    require(not emp.empty and not emp[["subject", "visit"]].isna().any().any(), "Empatica summary needs nonempty subject/visit keys.")
    require(emp.subject.str.fullmatch(r"[0-9]+").all(), "Subject identifiers must be numeric text.")
    require(not emp.duplicated(["subject", "visit"]).any(), 'Duplicate Empatica subject/visit keys.')
    require(set(emp.visit) == {"T0"}, 'Only T0 reports are supported.')
    records = {(r["subject"], r["visit"]): r for r in status["recordings"]}
    require(len(records) == len(status["recordings"]), "Duplicate McRoberts subject/visit keys.")
    require(set(zip(emp_intervals.subject, emp_intervals.visit)) <= set(zip(emp.subject, emp.visit)),
            "Empatica intervals contain subjects/visits absent from its summary.")
    require(set(records) == set(zip(emp.subject, emp.visit)), "Cohorts differ; audit required.")
    subjects, interval_rows, state_rows, daily_rows = [], [], [], []
    max_emp_error = max_mc_wear_error = max_mc_observed_error = 0.0
    for row in emp.itertuples(index=False):
        subject = row.subject
        require(records[(subject, row.visit)]["status"] == "completed", 'McRoberts recording is incomplete.')
        lo, hi = pd.Timestamp(row.recording_start).value, pd.Timestamp(row.protocol_end).value
        require(all(pd.notna(pd.Timestamp(v)) and pd.Timestamp(v).tz is None
                    for v in (row.recording_start, row.protocol_end)), "Protocol timestamps must be timezone-naive.")
        require(hi - lo == PROTOCOL and row.protocol_duration_hours == 168, 'Each shared protocol window must span exactly 168 hours.')
        full = [[lo, hi]]
        ei = emp_intervals.loc[emp_intervals.subject.eq(subject)]
        require(set(ei.category) <= {"detected_nonwear", "recording_gap", "merge_padding", "missing_recording"}, 'Unknown Empatica interval category.')
        emp_excluded = read_intervals(ei)
        require(all(lo <= s < e <= hi for s, e in emp_excluded), 'Empatica exclusion interval lies outside its protocol window.')
        require(abs(hours(emp_excluded) - ei.duration_hours.sum()) < 1e-8, 'Empatica exclusion intervals overlap or duration totals disagree.')
        ew = subtract(full, emp_excluded)
        emp_error = abs(hours(ew) - row.not_masked_hours) * 3600
        require(abs(100*hours(ew)/168 - row.not_masked_percent) < 1e-8, "Empatica source percentage does not reconcile.")
        max_emp_error = max(max_emp_error, emp_error)
        require(emp_error < 1e-5, (subject, "Empatica source mismatch", emp_error))

        folder = mcroberts_run / "silver" / subject / row.visit / "McRoberts" / "nonwear"
        ms = load_json(folder / "nonwear_summary.json")
        quality = load_json(folder / "timestamp_quality.json")
        nw = read_intervals(load_csv(folder / "final_nonwear_intervals.csv"))
        invalid = read_intervals(load_csv(folder / "time_invalid_intervals.csv"))
        mc_start = pd.Timestamp(ms["start"]).value
        require(np.isfinite(ms["sample_rate_hz"]) and ms["sample_rate_hz"] > 0,
                "McRoberts sample rate must be positive and finite.")
        require(all(pd.notna(pd.Timestamp(v)) and pd.Timestamp(v).tz is None
                    for v in (ms["start"], ms["last_sample"])), "McRoberts timestamps must be timezone-naive.")
        require(0 <= ms["revised_nonwear_hours"] <= ms["observed_hours"] <= 168 + 1/3600,
                "McRoberts nonwear and observation hours are inconsistent.")
        # A sample occupies [timestamp, timestamp + 1/fs). Enforce original elapsed cap.
        mc_end = min(pd.Timestamp(ms["last_sample"]).value + round(1e9 / ms["sample_rate_hz"]), mc_start + PROTOCOL)
        mc_span = [[mc_start, mc_end]]
        observed = subtract(mc_span, invalid)
        mc_full_wear = subtract(observed, nw)
        observed_error = abs(hours(observed) - ms["observed_hours"]) * 3600
        wear_error = abs(hours(mc_full_wear) - (ms["observed_hours"] - ms["revised_nonwear_hours"])) * 3600
        max_mc_observed_error = max(max_mc_observed_error, observed_error)
        max_mc_wear_error = max(max_mc_wear_error, wear_error)
        require(max(observed_error, wear_error) < 1, (subject, "McRoberts source mismatch", observed_error, wear_error))
        mw = clipped(mc_full_wear, lo, hi)
        joint = intersect(mw, ew)
        mc_nw = clipped(intersect(observed, nw), lo, hi)
        mc_gap = clipped(invalid, lo, hi)
        mc_outside = subtract(full, mc_span)
        states = partition(mw, ew, lo, hi)
        totals = Counter()
        for state, start, end in states:
            totals[state] += end - start
            state_rows.append([subject, row.visit, state, stamp(start), stamp(end), (end-start)/NS_HOUR])
        require(sum(totals.values()) == PROTOCOL, 'Coverage states do not conserve the protocol window.')
        require(totals["both"] == sum(e-s for s, e in joint), 'Coverage sweep and interval intersection disagree.')
        require(totals["both"] + totals["mcroberts_only"] == sum(e-s for s, e in mw), 'Coverage sweep and interval intersection disagree.')
        require(totals["both"] + totals["empatica_only"] == sum(e-s for s, e in ew), 'Coverage sweep and interval intersection disagree.')
        require(abs(hours(mw) + hours(mc_nw) + hours(mc_gap) + hours(mc_outside) - 168) < 1e-7, 'McRoberts wear, nonwear, missing and outside time do not partition the window.')
        require(hours(joint) <= min(hours(mw), hours(ew)) + 1e-10, 'McRoberts wear, nonwear, missing and outside time do not partition the window.')
        for metric, intervals in (("mcroberts", mw), ("empatica", ew), ("both", joint)):
            for start, end in intervals:
                interval_rows.append([subject, row.visit, metric, stamp(start), stamp(end), (end-start)/NS_HOUR])
        before = max(0, min(mc_start, hi)-lo) / NS_HOUR
        after = max(0, hi-max(mc_end, lo)) / NS_HOUR
        analysis_n = quality["analysis"]["n_samples"]
        source_n = next((r["retained_samples"] for r in quality.get("streams", []) if r["stream"] == "a"), None)
        # Only label an end as cap-censored when it reaches the cap and extra source samples exist.
        cap_reached = (source_n is not None and abs(mc_end - (mc_start + PROTOCOL)) < 1e9
                       and source_n > analysis_n)
        cap_tail = after if cap_reached else 0.0
        joint_cap_unknown = hours(intersect([[max(mc_end, lo), hi]], ew)) if cap_tail else 0.0
        flags = []
        if after > 0 and source_n is None:
            flags.append("mcroberts_tail_cap_status_unknown")
        if before > 0:
            flags.append("mcroberts_starts_after_shared_window")
        if after > 0:
            flags.append("mcroberts_cap_before_shared_end" if cap_reached else "mcroberts_recording_ends_before_shared_end")
        if row.temperature_alignment_warnings:
            flags.append("empatica_temperature_alignment_warning")
        if row.unassessed_not_masked_hours > 0:
            flags.append("empatica_includes_detach_unassessed_time")
        out = dict(subject=subject, visit=row.visit, protocol_start=stamp(lo), protocol_end=stamp(hi), protocol_hours=168)
        for metric, intervals in (("mcroberts", mw), ("empatica", ew), ("both", joint)):
            out[f"{metric}_wear_hours"] = hours(intervals)
            out[f"{metric}_wear_percent"] = 100 * hours(intervals) / 168
        for state in ("mcroberts_only", "empatica_only", "neither"):
            out[f"{state}_hours"] = totals[state] / NS_HOUR
        out.update(
            mcroberts_nonwear_hours=hours(mc_nw), mcroberts_time_invalid_hours=hours(mc_gap),
            mcroberts_outside_processed_hours=hours(mc_outside), mcroberts_before_processed_hours=before,
            mcroberts_after_processed_hours=after, mcroberts_cap_censored_tail_hours=cap_tail,
            both_cap_censored_potential_hours=joint_cap_unknown,
            mcroberts_wear_percent_upper_if_censored_tail_worn=100*(hours(mw)+cap_tail)/168,
            both_wear_percent_upper_if_censored_tail_worn=100*(hours(joint)+joint_cap_unknown)/168,
            mcroberts_recording_start=stamp(mc_start), mcroberts_processed_end=stamp(mc_end),
            mcroberts_start_offset_seconds=(mc_start-lo)/1e9,
            mcroberts_own_window_observed_hours=ms["observed_hours"],
            mcroberts_own_window_nonwear_hours=ms["revised_nonwear_hours"],
            empatica_detected_nonwear_hours=row.detected_nonwear_hours,
            empatica_recording_gap_hours=row.recording_gap_hours, empatica_merge_padding_hours=row.merge_padding_hours,
            empatica_missing_recording_hours=row.missing_recording_hours,
            empatica_detach_unassessed_not_masked_hours=row.unassessed_not_masked_hours,
            empatica_temperature_alignment_warnings=int(row.temperature_alignment_warnings),
            empatica_short_gap_hours_full_source_qc=row.source_short_recording_gaps_1_to_60s_hours,
            qc_flags="; ".join(flags))
        subjects.append(out)
        for day in range(7):
            dstart, dend = lo+day*24*NS_HOUR, lo+(day+1)*24*NS_HOUR
            d = dict(subject=subject, visit=row.visit, protocol_day=day+1, start=stamp(dstart), end=stamp(dend), day_hours=24)
            for metric, intervals in (("mcroberts", mw), ("empatica", ew), ("both", joint)):
                d[f"{metric}_wear_hours"] = hours(clipped(intervals, dstart, dend))
                d[f"{metric}_wear_percent_of_day"] = d[f"{metric}_wear_hours"] / 24 * 100
            daily_rows.append(d)

    for path, record in sources.items():
        require(hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], "Source changed during report calculation.")
    output.mkdir(parents=True, exist_ok=False)
    subjects = pd.DataFrame(subjects)
    subjects.to_csv(output / "subject_summary.csv", index=False)
    pd.DataFrame(interval_rows, columns=["subject", "visit", "measure", "start", "end", "duration_hours"]).to_csv(output / "wear_intervals.csv", index=False)
    pd.DataFrame(state_rows, columns=["subject", "visit", "coverage_state", "start", "end", "duration_hours"]).to_csv(output / "coverage_states.csv", index=False)
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(output / "daily_summary.csv", index=False)
    for metric in METRICS:
        require(np.allclose(daily.groupby("subject")[f"{metric}_wear_hours"].sum().reindex(subjects.subject), subjects[f"{metric}_wear_hours"], rtol=0, atol=1e-9), 'Daily wear does not reconcile with subject totals.')
    summary, thresholds, bands = [], [], []
    for metric in METRICS:
        x = subjects[f"{metric}_wear_percent"]
        summary.append(dict(measure=metric, label=LABELS[metric], n_subjects=len(x), n_any_wear=int((x>0).sum()),
                            mean_percent=x.mean(), sd_percent=x.std(), median_percent=x.median(), q25_percent=x.quantile(.25),
                            q75_percent=x.quantile(.75), min_percent=x.min(), max_percent=x.max(),
                            total_wear_hours=subjects[f"{metric}_wear_hours"].sum(), median_wear_hours=subjects[f"{metric}_wear_hours"].median()))
        for threshold in (0, 25, 50, 70, 75, 80, 90, 95, 99, 100):
            n = int((x >= threshold).sum())
            thresholds.append(dict(measure=metric, threshold_percent=threshold, minimum_wear_hours=threshold/100*168, n_subjects=n, percent_subjects=n/len(x)*100))
        for left, right in ((0,25),(25,50),(50,75),(75,90),(90,95),(95,100)):
            count = int(((x>=left) & ((x<=right) if right==100 else (x<right))).sum())
            bands.append(dict(measure=metric, coverage_band=f"{left}–{'≤' if right==100 else '<'}{right}%", lower_percent=left, upper_percent=right, n_subjects=count, percent_subjects=count/len(x)*100))
        require(sum(r["n_subjects"] for r in bands if r["measure"]==metric) == len(x), 'Coverage bands do not conserve the cohort.')
    summary, thresholds, bands = pd.DataFrame(summary), pd.DataFrame(thresholds), pd.DataFrame(bands)
    summary.to_csv(output / "cohort_summary.csv", index=False)
    thresholds.to_csv(output / "coverage_thresholds.csv", index=False)
    bands.to_csv(output / "coverage_bands.csv", index=False)
    validation = dict(n_subjects=len(subjects), matched_subjects=len(records), all_source_statuses_completed=True,
                      identical_168h_shared_windows=True, interval_sweep_matches_intersection=True,
                      four_state_partition_conserves_168h=True, daily_totals_reconcile=True,
                      empatica_max_source_difference_seconds=max_emp_error,
                      mcroberts_max_own_window_observed_difference_seconds=max_mc_observed_error,
                      mcroberts_max_own_window_wear_difference_seconds=max_mc_wear_error,
                      mcroberts_cap_censored_subjects=int((subjects.mcroberts_cap_censored_tail_hours>0).sum()),
                      mcroberts_cap_censored_total_hours=float(subjects.mcroberts_cap_censored_tail_hours.sum()),
                      max_mcroberts_cap_censored_minutes=float(subjects.mcroberts_cap_censored_tail_hours.max()*60))
    validation["cap_censoring_threshold_sensitivity"] = {
        m: {str(t): int(((subjects[f"{m}_wear_percent"] < t) & (subjects[f"{m}_wear_percent_upper_if_censored_tail_worn"] >= t)).sum())
            for t in (25, 50, 70, 75, 80, 90, 95, 99, 100)} for m in ("mcroberts", "both")}
    (output / "validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    manifest = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), window="Empatica recording_start to protocol_end, 168 stored-timestamp hours", mcroberts_run=str(mcroberts_run), empatica_dir=str(empatica_dir), inputs=[])
    load_bytes(Path(__file__).resolve())
    manifest["inputs"] = [sources[path] for path in sorted(sources)]
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2)+"\n")
    notes = [
        f"T0 wear coverage across {len(subjects)} matched participants. No clinical exclusions were applied.",
        "All three measures use the Empatica report's subject-specific [recording_start, protocol_end) window and a fixed 168-hour denominator.",
        "Wear means pipeline-defined estimated wear coverage. It does not establish physiological signal quality or verified physical wearing.",
        "McRoberts: processed span minus time-invalid intervals and final respiration-assisted nonwear. The posture/TIB-specific lying gap fill is not used.",
        "McRoberts sample support ends one sample period after last_sample and is clipped to the original elapsed 168-hour cap. Interval timestamps use half-open [start,end) bounds.",
        "Empatica: complement of detected_nonwear, recording_gap, merge_padding and missing_recording intervals. Hours and percentages reproduce the supplied not_masked_hours and not_masked_percent.",
        "Empatica may include recording holes of 60 seconds or less, DETACH-unassessed chunks and poor physiological signal, consistently with the existing report. Short-gap QC values describe the full original source, not only the protocol window, and are not subtracted.",
        "Both simultaneously: exact intersection of the timestamped McRoberts and Empatica wear intervals. Device timestamps are not shifted to force agreement.",
        "Outside McRoberts processed coverage is unavailable, not detected nonwear. A saved McRoberts cap can end before the shared window even when additional raw source data exist. Cap-censored tail time is reported explicitly and contributes no assumed wear.",
        f"Cap-censored tails affect {validation['mcroberts_cap_censored_subjects']} participants, totalling {validation['mcroberts_cap_censored_total_hours']:.6f} hours (maximum {validation['max_mcroberts_cap_censored_minutes']:.3f} minutes per participant). Results use saved processed coverage; underlying wear in those tails is unknown.",
        "Subject columns give upper bounds if all cap-censored McRoberts tail time were wear. The simultaneous upper bound adds only censored time overlapping Empatica wear. These bounds do not address other unavailable time or sensor classification error.",
        "The four states both, mcroberts_only, empatica_only and neither partition the shared window. Neither means neither has estimated wear coverage; it does not prove neither device was physically worn.",
        "Source timestamps have no timezone. Stored timestamp differences are retained, consistently with the original reports and Europe/Rome study configuration; no extra DST conversion or clock synchronization is introduced.",
        "Thresholds are descriptive, cumulative >= cutoffs on unrounded percentages. No clinical validity or minimum valid-day filter is applied. Coverage bands are mutually exclusive.",
        "Daily rows are seven successive 24-hour periods from the common protocol start, not calendar days. IDs are text and preserve leading zeros.",
        "Validation: Empatica source totals reproduced, McRoberts original-window totals reconciled within one second, exact endpoint sweep independently verifies the intersection and four-state partition, and daily totals reconcile.",
    ]
    (output / "METHODS.txt").write_text("\n\n".join(notes)+"\n")
    (output / "workbook_data.json").write_text(json.dumps(dict(subjects=subjects.to_dict("records"), summary=summary.to_dict("records"), thresholds=thresholds.to_dict("records"), bands=bands.to_dict("records"), notes=notes, validation=validation, sources=dict(empatica=str(empatica_dir), mcroberts=str(mcroberts_run))), indent=2))
    if plots:
        make_plots(subjects, thresholds, output)
    method_html="".join(f"<li>{html.escape(n)}</li>" for n in notes)
    display = summary[["label", "n_subjects", "n_any_wear", "median_percent", "q25_percent", "q75_percent", "mean_percent"]]
    counts=thresholds.pivot(index="threshold_percent",columns="measure",values="n_subjects").rename(columns=LABELS)
    plot_html = '<img src="wear_coverage.png" alt="Wear coverage distributions and cumulative participant counts">' if plots else ""
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>T0 wear coverage</title><style>body{{font:16px Arial,sans-serif;max-width:1100px;margin:40px auto;color:#172d40;line-height:1.5}}h1,h2{{font-weight:600}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{padding:9px;text-align:right;border-bottom:1px solid #dae1e7}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf2f7}}img{{width:100%}}li{{margin-bottom:10px}}a{{color:#185b96}}</style><h1>T0 wear coverage over seven days</h1><p>{len(subjects)} matched participants. All measures use the same subject-specific 168-hour Empatica protocol window.</p><h2>Coverage distribution (%)</h2>{display.to_html(index=False,float_format=lambda x:f'{x:.2f}')}{plot_html}<h2>Participants meeting each threshold</h2><p>Thresholds are cumulative and use unrounded percentages.</p>{counts.to_html()}<p><a href="subject_summary.csv">Subject results CSV</a> · <a href="wear_intervals.csv">Wear intervals CSV</a> · <a href="daily_summary.csv">Daily results CSV</a></p><h2>Definitions and limitations</h2><ol>{method_html}</ol></html>'''
    (output / "report.html").write_text(document)
    print(summary.to_string(index=False))
    print(thresholds.loc[thresholds.threshold_percent.isin([50,75,90,95,99])].to_string(index=False))
    print(json.dumps(validation, indent=2))


def make_plots(subjects, thresholds, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#1D6096", "#D47B2C", "#347C63"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    series = [subjects[f"{m}_wear_percent"].to_numpy() for m in METRICS]
    boxes = axes[0].boxplot(series, patch_artist=True, widths=.45, showfliers=False,
                            medianprops=dict(color="white", linewidth=2))
    rng = np.random.default_rng(27)
    for i, (values, color, box) in enumerate(zip(series, colors, boxes["boxes"])):
        box.set_facecolor(color)
        axes[0].scatter(i+1+rng.uniform(-.14,.14,len(values)), values, s=9, alpha=.24, c=color, zorder=1)
    axes[0].set_xticks([1,2,3], ["McRoberts", "Empatica", "Both\nsimultaneously"])
    axes[0].set_ylabel("Coverage (% of the same 168 hours)")
    axes[0].set_ylim(0,103)
    axes[0].set_title("Participant coverage distributions")
    for metric, color in zip(METRICS, colors):
        values = subjects[f"{metric}_wear_percent"].to_numpy()
        cuts = np.r_[0, np.unique(values), 100]
        counts = np.array([(values>=cut).sum() for cut in cuts])
        axes[1].step(cuts, counts, where="pre", color=color, lw=2, label=LABELS[metric])
    axes[1].set_xlim(0,100)
    axes[1].set_ylim(0, max(1, len(subjects))*1.03)
    axes[1].set_xlabel("Minimum wear coverage (% of 168 hours)")
    axes[1].set_ylabel("Participants meeting threshold (n)")
    axes[1].set_title("Cumulative coverage thresholds")
    axes[1].legend(frameon=False, loc="lower left")
    for ax in axes:
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"T0 wear coverage · {len(subjects)} matched participants", fontsize=15)
    fig.savefig(output/"wear_coverage.png",dpi=180)
    fig.savefig(output/"wear_coverage.svg")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empatica-dir", required=True, type=Path,
                        help="Saved Empatica subject_summary.csv and intervals.csv directory.")
    parser.add_argument("--mcroberts-run", required=True, type=Path,
                        help="Completed full lower-back run containing run_status.json and silver/.")
    parser.add_argument("--output", required=True, type=Path, help="New report directory; must not exist.")
    parser.add_argument("--no-plots", action="store_true", help="Write tables and HTML without figures.")
    args = parser.parse_args(argv)
    analyze(args.empatica_dir.expanduser().resolve(), args.mcroberts_run.expanduser().resolve(),
            args.output.expanduser().resolve(), plots=not args.no_plots)


if __name__ == "__main__":
    main()
