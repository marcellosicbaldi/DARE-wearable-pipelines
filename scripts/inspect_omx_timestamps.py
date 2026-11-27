"""Read-only packet-header diagnosis of failed McRoberts OMX recordings.

Uses the current reader's timestamp convention, without loading raw sensor
samples or repairing timestamps. Outputs contain private recording dates/IDs;
choose a private output root. Packet numbers in reports are zero based.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time
from uuid import uuid4

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from dare_wearables.common.output_state import signature
from dare_wearables.lower_back.io.mcroberts_loader import packet_sample_times_vector

HEADER = np.dtype({
    "names": ["type", "stream", "length", "sequence", "timestamp", "fraction", "offset", "rate", "modifier", "data_type", "conversion", "packing", "count"],
    "formats": ["S1", "S1", "<u2", "<u4", "<u4", "<u2", "<i2", "<u2", "i1", "u1", "i1", "u1", "<u2"],
    "offsets": [0, 1, 2, 4, 8, 12, 14, 16, 18, 19, 20, 21, 22], "itemsize": 512})


def _times(header):
    """Vectorized equivalent of the current packed-date packet decoder."""
    packed = header["timestamp"].astype(np.int64)
    month = ((packed >> 22) & 15) - 1 + ((packed >> 26) & 63) * 12
    date = (np.datetime64("2000-01", "M") + month.astype("timedelta64[M]")).astype("datetime64[D]")
    second = (((packed >> 12) & 31) * 3600 + ((packed >> 6) & 63) * 60 + (packed & 63))
    base = date.astype("datetime64[ns]").astype(np.int64) + (((packed >> 17) & 31) - 1) * 86400_000_000_000 + second * 1_000_000_000
    rate = header["rate"].astype(float)
    modifier = header["modifier"].astype(float)
    fs = np.where(modifier > 1, rate / np.maximum(modifier, 1),
                  np.where(modifier < -1, -modifier * rate,
                           np.where(modifier == 0, 1 / rate,
                                    np.where(modifier == -1, 1 / (rate * 60), rate))))
    if np.any(rate == 0):
        raise ValueError("zero packet sample rate")
    starts = base + (header["fraction"].astype(float) * 1e9 / 65536).astype(np.int64) - (header["offset"] * 1e9 / fs).astype(np.int64)
    ends = starts + ((header["count"] - 1) * 1e9 / fs).astype(np.int64)
    return starts, ends, fs


def _iso(value):
    return str(np.datetime64(int(value), "ns"))


def inspect_file(source, *, label=None):
    source = Path(source)
    before = signature({}, [source])
    packets = np.memmap(source, dtype=HEADER, mode="r", shape=(source.stat().st_size // 512,))
    streams, events, contexts = [], [], []
    for sid in (b"a", b"g"):
        ids = np.flatnonzero((packets["stream"] == sid) & (packets["count"] > 0) & (packets["count"] <= 80))
        h = {name: packets[name][ids].copy() for name in HEADER.names}
        if not len(ids):
            streams.append({"stream": sid.decode(), "packet_count": 0})
            continue
        starts, ends, fs = _times(h)
        # Verify the vectorized decoder against the production implementation.
        checks = np.unique(np.r_[np.linspace(0, len(ids)-1, min(20, len(ids)), dtype=int), np.flatnonzero(starts[1:] <= ends[:-1]) + 1])
        for i in checks:
            actual = packet_sample_times_vector(int(h["timestamp"][i]), int(h["fraction"][i]), int(h["offset"][i]), int(h["rate"][i]), int(h["modifier"][i]), int(h["count"][i]))
            np.testing.assert_array_equal(actual[[0, -1]].astype("datetime64[ns]").astype(np.int64), [starts[i], ends[i]])
        delta = starts[1:] - ends[:-1]
        expected = 1e9 / fs[1:]
        bad = np.flatnonzero(delta <= 0) + 1
        forward = np.flatnonzero(delta - expected > 50_000) + 1
        all_discontinuities = np.flatnonzero(np.abs(delta - expected) > 50_000) + 1
        seq_delta = np.diff(h["sequence"].astype(np.int64))
        # Difference against an uninterrupted sample-count timeline. Stable
        # rates/sequence counters are reported separately, not assumed repaired.
        positions = np.r_[0, np.cumsum(h["count"][:-1].astype(np.int64))]
        residual = starts - starts[0] - (positions * 1e9 / fs[0]).astype(np.int64)
        for i in bad:
            recovery = np.flatnonzero(starts[i:] > ends[i-1])
            j = int(i + recovery[0]) if len(recovery) else None
            # A nearby positive jump can indicate a temporary bad timestamp
            # anchor, rather than a permanent change to the clock timeline.
            following = all_discontinuities[all_discontinuities > i]
            next_jump = int(following[0]) if len(following) else None
            row = {"stream": sid.decode(), "stream_packet_index": int(i), "file_packet": int(ids[i]),
                   "sequence": int(h["sequence"][i]), "sequence_delta": int(seq_delta[i-1]),
                   "previous_last_sample": _iso(ends[i-1]), "first_sample": _iso(starts[i]),
                   "boundary_delta_seconds": float(delta[i-1]/1e9),
                   "timeline_step_seconds": float((delta[i-1]-expected[i-1])/1e9),
                   "header_timestamp_previous": int(h["timestamp"][i-1]), "header_timestamp": int(h["timestamp"][i]),
                   "previous_offset_samples": int(h["offset"][i-1]), "offset_samples": int(h["offset"][i]),
                   "recovery_stream_packet_index": j, "packets_before_wall_time_recovers": j-int(i) if j is not None else None,
                   "next_discontinuity_packet": int(ids[next_jump]) if next_jump is not None else None,
                   "packets_until_next_discontinuity": next_jump-int(i) if next_jump is not None else None,
                   "next_discontinuity_step_seconds": float((delta[next_jump-1]-expected[next_jump-1])/1e9) if next_jump is not None else None,
                   "residual_before_seconds": float(residual[i-1]/1e9), "residual_after_seconds": float(residual[i]/1e9),
                   "residual_after_next_discontinuity_seconds": float(residual[next_jump]/1e9) if next_jump is not None else None}
            events.append(row)
            context_ids = set(range(max(0, int(i)-4), min(len(ids), int(i)+5)))
            if next_jump is not None:
                context_ids.update(range(max(0,next_jump-2),min(len(ids),next_jump+3)))
            for k in sorted(context_ids):
                packet_bytes = np.memmap(source, mode="r", dtype="u1", offset=int(ids[k])*512, shape=(512,))
                checksum_valid = int(np.frombuffer(packet_bytes, dtype="<u2").sum()) % 65536 == 0
                contexts.append({"stream": sid.decode(), "event_packet": int(ids[i]), "file_packet": int(ids[k]),
                                 **{name: int(h[name][k]) for name in ("sequence","timestamp","fraction","offset","rate","modifier","count","data_type","conversion","packing")},
                                 "first_sample": _iso(starts[k]), "last_sample": _iso(ends[k]),
                                 "boundary_delta_seconds": float((starts[k]-ends[k-1])/1e9) if k else None,
                                 "checksum_valid": checksum_valid})
        streams.append({"stream": sid.decode(), "packet_count": len(ids), "sample_count": int(h["count"].sum()),
                        "start": _iso(starts[0]), "end_in_file_order": _iso(ends[-1]),
                        "nonincreasing_boundaries": len(bad), "forward_steps_over_50us": len(forward),
                        "all_steps_over_50us": len(all_discontinuities), "sequence_discontinuities": int(np.count_nonzero(seq_delta != 1)),
                        "sequence_nonincreasing": int(np.count_nonzero(seq_delta <= 0)),
                        "sample_rates": np.unique(fs).tolist(), "packing": np.unique(h["packing"]).tolist(),
                        "non_data_packet_type": int(np.count_nonzero(h["type"] != b"d")),
                        "unexpected_payload_length": int(np.count_nonzero(h["length"] != 508)),
                        "serial_timestamp_mode_count": int(np.count_nonzero(h["timestamp"] & 0x80000000)),
                        "timeline_residual_min_seconds": float(residual.min()/1e9),
                        "timeline_residual_max_seconds": float(residual.max()/1e9),
                        "timeline_residual_final_seconds": float(residual[-1]/1e9)})
    if before != signature({}, [source]):
        raise ValueError("Input changed during the read-only audit")
    return {"label": label, "source": str(source.resolve()), "input_identity": before,
            "trailing_bytes": source.stat().st_size % 512, "streams": streams,
            "events": events, "packet_context": contexts}


def run(assessment, output_root, *, controls=10, seed=20261102):
    manifest = json.loads((Path(assessment) / "assessment.json").read_text())
    failures = [i for i in manifest["recordings"] if i["status"] == "failed"]
    good = sorted((i for i in manifest["recordings"] if i["status"] == "completed"), key=lambda i:(i["subject"],i["visit"]))
    chosen = [*(dict(i, audit_group="failed") for i in failures),
              *(dict(i, audit_group="control") for i in random.Random(seed).sample(good, min(controls,len(good))))]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(output_root) / f"timestamps_{stamp}_{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    (root/"selection.json").write_text(json.dumps({"assessment":str(assessment),"seed":seed,"recordings":chosen},indent=2,default=str)+"\n")
    print(f"Audit: {root}",flush=True)
    reports=[]
    for item in chosen:
        tick=time.monotonic()
        report=inspect_file(item["source"],label=f"{item['subject']}/{item['visit']}")
        report.update(subject=item["subject"],visit=item["visit"],audit_group=item["audit_group"])
        (root/f"{item['subject']}_{item['visit']}.json").write_text(json.dumps(report,indent=2,default=str)+"\n")
        reports.append(report)
        print(f"{report['label']} ({item['audit_group']}): " + "; ".join(f"{s['stream']}: {s.get('nonincreasing_boundaries',0)} backward/repeated boundaries" for s in report['streams'])+f"; {time.monotonic()-tick:.1f}s",flush=True)
        rows=[{k:r[k] for k in ("subject","visit","audit_group")} | s for r in reports for s in r["streams"]]
        pd.DataFrame(rows).to_csv(root/"stream_summary.csv",index=False)
        pd.DataFrame([{k:r[k] for k in ("subject","visit","audit_group")} | e for r in reports for e in r["events"]]).to_csv(root/"backward_events.csv",index=False)
        pd.DataFrame([{k:r[k] for k in ("subject","visit","audit_group")} | e for r in reports for e in r["packet_context"]]).to_csv(root/"packet_context.csv",index=False)
    (root/"audit.json").write_text(json.dumps({"status":"completed","recordings":len(reports),"repaired":False},indent=2)+"\n")
    return root


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--controls",type=int,default=10)
    parser.add_argument("--seed",type=int,default=20261102)
    args=parser.parse_args()
    run(args.assessment,args.output_root,controls=args.controls,seed=args.seed)
