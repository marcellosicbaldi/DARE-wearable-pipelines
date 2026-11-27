"""Diary import and cached, time-preserving raw signal views for manual review."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import tomllib

import numpy as np
import pandas as pd

from dare_wearables.common.output_state import _digest, signature
from dare_wearables.lower_back.io import mcroberts_loader
from dare_wearables.lower_back.io.timestamp_quality import TimestampOptions
from dare_wearables.redcap.identity import normalize_subjects

CHANNELS = ('ax', 'ay', 'az', 'gx', 'gy', 'gz')
PAIRS = (('no_sensor_time_2', 'no_sensor_time_11'),
         ('no_sensor_time_3', 'no_sensor_time_12'),
         ('no_sensor_time_6', 'no_sensor_time_13'),
         ('no_sensor_time_10', 'no_sensor_time_14'))


def subject_id(value):
    return str(normalize_subjects(pd.Series([value])).iloc[0])


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(handle, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@dataclass(frozen=True)
class AnnotationConfig:
    bronze_root: Path
    diary_csv: Path
    output_root: Path
    cache_root: Path
    subject: str = '900001'
    visit: str = 'T0'
    timestamp_policy: str = 'quarantine_bad_anchors'
    protocol_hours: float = 168.
    reviewer: str = ''

    @classmethod
    def from_toml(cls, path):
        path = Path(path).expanduser().resolve()
        with path.open('rb') as stream:
            values = tomllib.load(stream)['annotation']
        allowed = set(cls.__dataclass_fields__)
        if values.keys() - allowed:
            raise ValueError(f'Unknown annotation settings: {sorted(values.keys() - allowed)}')
        for name in ('bronze_root', 'diary_csv', 'output_root', 'cache_root'):
            item = Path(values[name]).expanduser()
            values[name] = (item if item.is_absolute() else path.parent / item).resolve()
        result = cls(**values)
        subject_id(result.subject)
        if not re.fullmatch(r'T\d+', result.visit):
            raise ValueError('visit must be a protocol visit such as T0.')
        TimestampOptions(policy=result.timestamp_policy).validate()
        if not np.isfinite(result.protocol_hours) or not 0 < result.protocol_hours <= 168:
            raise ValueError('protocol_hours must be in (0, 168].')
        for folder in (result.output_root, result.cache_root):
            if folder == result.bronze_root or result.bronze_root in folder.parents:
                raise ValueError('Annotation outputs and cache must be outside bronze_root.')
            if folder == result.diary_csv or folder in result.diary_csv.parents:
                raise ValueError('Choose an output/cache directory separate from the diary source.')
        return result


class Diary:
    """Only waist/lower-back fields, with the study's agreed calendar rules."""

    def __init__(self, path):
        self.path = Path(path)
        names = {'patient_id', 'redcap_event_name'}
        for day in range(1, 8):
            suffix = '' if day == 1 else f'_g{day}'
            names.update(name + suffix for name in ('encounter_ds', 'no_sensor', *(v for pair in PAIRS for v in pair)))
        data = pd.read_csv(self.path, dtype=str, keep_default_na=False, usecols=lambda name: name in names)
        if missing := names - set(data):
            raise ValueError(f'Diary is missing lower-back fields: {sorted(missing)}')
        data = data.loc[data.redcap_event_name == 'baseline_arm_1'].copy()
        data['subject'] = normalize_subjects(data.patient_id)
        if data.subject.duplicated().any():
            raise ValueError('Expected one baseline diary per participant.')
        self.rows = data.set_index('subject')
        self.sha256 = _digest(self.path)

    def intervals(self, subject):
        row = self.rows.loc[subject_id(subject)]
        intervals, issues = [], []
        for day in range(1, 8):
            suffix = '' if day == 1 else f'_g{day}'
            date_text, count_text = row['encounter_ds' + suffix].strip(), row['no_sensor' + suffix].strip()
            if count_text == '':
                count = 0  # Study instruction: no reported periods; does not establish wear.
            elif re.fullmatch('[0-4](?:\.0)?', count_text):
                count = int(float(count_text))
            else:
                issues.append(f'Day {day}: invalid period count; entries excluded.')
                continue
            day_date = pd.NaT
            for pattern, fmt in ((r'\d{4}-\d{2}-\d{2}', '%Y-%m-%d'),
                                 (r'\d{2}/\d{2}/\d{4}', '%d/%m/%Y')):
                if re.fullmatch(pattern, date_text):
                    day_date = pd.to_datetime(date_text, format=fmt, errors='coerce')
            for slot, (a, b) in enumerate(PAIRS, 1):
                start_text, end_text = row[a + suffix].strip(), row[b + suffix].strip()
                if slot > count:
                    if start_text or end_text:
                        issues.append(f'Day {day}, period {slot}: outside declared count; excluded.')
                    continue
                if pd.isna(day_date) or not all(re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?', t)
                                               for t in (start_text, end_text)):
                    issues.append(f'Day {day}, period {slot}: missing/invalid date or endpoint; excluded.')
                    continue
                clocks = [pd.Timedelta(t if len(t) == 8 else t + ':00') for t in (start_text, end_text)]
                start, end = day_date + clocks[0], day_date + clocks[1]
                if end < start:
                    start -= pd.Timedelta(days=1)
                if end == start:
                    issues.append(f'Day {day}, period {slot}: zero duration; excluded.')
                    continue
                intervals.append({'id': f'diary_{day}_{slot}', 'start_ns': int(start.value), 'end_ns': int(end.value),
                                  'diary_day': day, 'period_slot': slot, 'diary_date': date_text,
                                  'start_field': a + suffix, 'end_field': b + suffix,
                                  'start_raw': start_text, 'end_raw': end_text})
        return sorted(intervals, key=lambda x: x['start_ns']), issues


def recording_path(config, subject):
    folder = config.bronze_root / subject_id(subject) / config.visit / 'McRoberts'
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.omx') if folder.is_dir() else []
    if len(paths) != 1:
        raise ValueError(f'Expected one OMX file for {subject}/{config.visit}; found {len(paths)}.')
    return paths[0]


def participants(config, diary):
    return sorted(p.name for p in config.bronze_root.iterdir()
                  if p.is_dir() and re.fullmatch(r'\d{4,}', p.name) and p.name in diary.rows.index
                  and (p / config.visit / 'McRoberts').is_dir())


def write_raw_cache(folder, frame, *, source, identity, fs, hours, policy):
    """Cache exact float64 raw values and int64 timestamps, never shifted/interpolated."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    times = pd.DatetimeIndex(frame.datetime).as_unit('ns').asi8
    if len(times) < 2 or not np.isfinite(fs) or fs <= 0:
        raise ValueError('Recording has no usable paired raw samples.')
    period = int(round(1e9 / fs))
    if np.any(np.diff(times) <= 0):
        raise ValueError('Raw timestamps must increase strictly.')
    protocol_end = int(times[0]) + round(hours * 3600 * 1e9)
    n = int(np.searchsorted(times, protocol_end))
    times = times[:n]
    np.save(folder / 'timestamps.npy', times, allow_pickle=False)
    signals = np.lib.format.open_memmap(folder / 'signals.npy', mode='w+', dtype='float64', shape=(n, 6))
    for first in range(0, n, 1_000_000):
        values = frame.iloc[first:min(n, first+1_000_000)][list(CHANNELS)].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError('Raw signals contain non-finite values.')
        signals[first:first+len(values)] = values
    signals.flush()
    cuts = np.flatnonzero((np.diff(times) > 1.5 * period) | (np.diff(times) < .5 * period)) + 1
    segments = np.column_stack((np.r_[0, cuts], np.r_[cuts, n]))
    gaps = [[int(times[a-1]) + period, int(times[b])] for (_, a), (b, _) in zip(segments[:-1], segments[1:])
            if times[b] > times[a-1] + period]
    bins, lo, hi, seg_ids = [], [], [], []
    for seg, (a, b) in enumerate(segments):
        starts = np.arange(a, b, max(1, round(15 * fs)), dtype=np.int64)
        stops = np.r_[starts[1:], b]
        bins.extend(np.column_stack((times[starts], times[stops-1])).tolist())
        lo.extend(np.minimum.reduceat(signals[a:b], starts-a, axis=0).tolist())
        hi.extend(np.maximum.reduceat(signals[a:b], starts-a, axis=0).tolist())
        seg_ids.extend([seg] * len(starts))
    np.savez(folder / 'overview.npz', bins=np.asarray(bins, dtype=np.int64), lo=np.asarray(lo), hi=np.asarray(hi),
             segments=np.asarray(seg_ids))
    meta = {'schema': 1, 'source': str(source), 'input_identity': identity, 'sample_rate_hz': float(fs),
            'n_samples': n, 'start_ns': int(times[0]), 'end_ns': min(protocol_end, int(times[-1]) + period),
            'segments': segments.tolist(), 'gaps': gaps, 'timestamp_policy': policy,
            'channels': list(CHANNELS), 'signals_dtype': 'float64',
            'source_first_sample': str(pd.Timestamp(times[0])), 'source_last_sample': str(pd.Timestamp(times[-1]))}
    atomic_json(folder / 'metadata.json', meta)
    del signals
    return meta


def prepare_cache(config, subject):
    source = recording_path(config, subject)
    identity = signature({}, [source])
    import dare_wearables.lower_back.io.timestamp_quality as quality
    parameters = {'cache_revision': 1, 'timestamp_policy': config.timestamp_policy, 'hours': config.protocol_hours,
                  'reader_sha256': _digest(mcroberts_loader.__file__), 'quality_sha256': _digest(quality.__file__)}
    key = signature(parameters, [source])
    folder = config.cache_root / key
    if (folder / 'metadata.json').is_file():
        cache = RawCache(folder)
        if cache.meta['input_identity'] != identity:
            raise ValueError('Cached recording identity does not match the OMX source.')
        return cache
    config.cache_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.preparing-', dir=config.cache_root))
    try:
        frame = mcroberts_loader.read_dp7_as_dataframe_fast(source, verbose=0,
                    timestamps=TimestampOptions(policy=config.timestamp_policy))
        fs = frame.attrs['sample_rate_hz']
        write_raw_cache(temporary, frame, source=source, identity=identity, fs=fs,
                        hours=config.protocol_hours, policy=config.timestamp_policy)
        if signature({}, [source]) != identity:
            raise ValueError('OMX source changed during loading.')
        del frame
        if folder.exists():
            # Another local process may have finished preparing this same source.
            cache = RawCache(folder)
            if cache.meta['input_identity'] != identity:
                raise ValueError('Conflicting cached recording.')
            return cache
        temporary.rename(folder)
        return RawCache(folder)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class RawCache:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / 'metadata.json').read_text())
        if self.meta.get('schema') != 1:
            raise ValueError('Unsupported raw cache format.')
        self.times = np.load(self.folder / 'timestamps.npy', mmap_mode='r', allow_pickle=False)
        self.values = np.load(self.folder / 'signals.npy', mmap_mode='r', allow_pickle=False)
        self.overview = dict(np.load(self.folder / 'overview.npz', allow_pickle=False))
        if (self.times.shape != (self.meta['n_samples'],) or self.times.dtype != np.dtype('int64')
                or self.values.shape != (len(self.times), 6) or self.values.dtype != np.dtype('float64')
                or self.times[0] != self.meta['start_ns']):
            raise ValueError('Incomplete or inconsistent raw cache; remove this cache folder and reload.')
        self.first_ns = int(self.times[0])
        self.end_ns = int(self.meta['end_ns'])
        self.fs = float(self.meta['sample_rate_hz'])
        self.duration = (self.end_ns-self.first_ns)/1e9

    def seconds(self, ns):
        return (int(ns)-self.first_ns)/1e9

    def boundary_ns(self, sample, edge='start'):
        if edge == 'end' and sample > 0:
            return min(self.end_ns, int(self.times[sample-1]) + round(1e9/self.fs))
        return self.end_ns if sample == len(self.times) else int(self.times[sample])

    def boundary(self, sample, edge='start'):
        ns = self.boundary_ns(sample,edge)
        return {'sample': int(sample), 'seconds': self.seconds(ns), 'time': pd.Timestamp(ns).isoformat()}

    def snap(self, seconds, edge='start'):
        if not isinstance(seconds, (int, float)) or not np.isfinite(seconds):
            raise ValueError('Boundary time must be finite.')
        target = self.first_ns + round(seconds * 1e9)
        if target < self.first_ns or target > self.end_ns:
            raise ValueError('Boundary is outside the recording.')
        index = int(np.searchsorted(self.times, target))
        candidates = [p for p in (index-1, index, index+1) if 0 <= p <= len(self.times)]
        sample = min(candidates, key=lambda p: abs(self.boundary_ns(p,edge)-target))
        if abs(self.boundary_ns(sample,edge)-target) > .51e9/self.fs + 1_000:
            raise ValueError('Boundary falls in missing/time-invalid data. Move it to an observed sample.')
        return self.boundary(sample,edge)

    def nearest(self, ns, edge='start'):
        """Seed a diary proposal at the nearest retained boundary, without confirming it."""
        ns = max(self.first_ns, min(self.end_ns, int(ns)))
        index = int(np.searchsorted(self.times, ns))
        return min((p for p in (index-1, index, index+1) if 0 <= p <= len(self.times)),
                   key=lambda p: abs(self.boundary_ns(p,edge)-ns))

    def view(self, left, right, width=1200):
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in (left, right)) or left >= right:
            raise ValueError('Choose an increasing finite view interval.')
        left, right = max(0., left), min(self.duration, right)
        if left >= right:
            raise ValueError('Plot interval is outside the recording.')
        width = max(200, min(2400, int(width)))
        a_ns, b_ns = self.first_ns+round(left*1e9), self.first_ns+round(right*1e9)
        a, b = int(np.searchsorted(self.times, a_ns)), int(np.searchsorted(self.times, b_ns))
        n = b-a
        if not n:
            return {'left': left, 'right': right, 'mode': 'empty', 'x': [], 'lo': [], 'hi': [],
                    'breaks': [], 'gaps': self.visible_gaps(a_ns,b_ns), 'samples': 0, 'bin_seconds': 0.}
        if right-left > 6*3600:
            overview = self.overview
            mask = (overview['bins'][:,1] >= a_ns) & (overview['bins'][:,0] < b_ns)
            bins, lo, hi = overview['bins'][mask], overview['lo'][mask], overview['hi'][mask]
            x_ns = bins[:,0] + (bins[:,1]-bins[:,0])//2
            ids = overview['segments'][mask]
            breaks = (np.flatnonzero(np.diff(ids) != 0)+1).tolist()
            mode, step = 'envelope', 15.
        else:
            x_parts, lows, highs, breaks = [], [], [], []
            points = 0
            raw_mode = n <= 30000
            stride = 1 if raw_mode else max(1, int(np.ceil(n/width)))
            for first, stop in self.meta['segments']:
                first, stop = max(a,first), min(b,stop)
                if first >= stop:
                    continue
                starts = np.arange(first, stop, stride)
                if points:
                    breaks.append(points)
                if raw_mode:
                    x_parts.append(self.times[starts])
                    lows.append(self.values[first:stop])
                else:
                    stops = np.r_[starts[1:],stop]
                    x_parts.append(self.times[starts] + (self.times[stops-1]-self.times[starts])//2)
                    lows.append(np.minimum.reduceat(self.values[first:stop], starts-first, axis=0))
                    highs.append(np.maximum.reduceat(self.values[first:stop], starts-first, axis=0))
                points += len(starts)
            x_ns = np.concatenate(x_parts)
            lo = np.concatenate(lows)
            hi = np.concatenate(highs) if highs else None
            mode, step = ('raw' if raw_mode else 'envelope'), stride/self.fs
        return {'left': left, 'right': right, 'mode': mode, 'x': ((x_ns-self.first_ns)/1e9).tolist(),
                'lo': lo.tolist(), 'hi': None if hi is None else hi.tolist(), 'breaks': breaks,
                'samples': n, 'bin_seconds': step, 'gaps': self.visible_gaps(a_ns,b_ns)}

    def visible_gaps(self, left, right):
        return [[self.seconds(a),self.seconds(b)] for a,b in self.meta['gaps'] if a < right and b > left]


def diary_for_recording(diary, subject, cache):
    source, issues = diary.intervals(subject)
    visible = [item for item in source if item['start_ns'] < cache.end_ns and item['end_ns'] > cache.first_ns]
    if len(source) != len(visible):
        issues.append(f'{len(source)-len(visible)} reported period(s) outside the recording; not displayed.')
    identity = hashlib.sha256(json.dumps({'source': source, 'issues': issues}, sort_keys=True).encode()).hexdigest()
    display, seeds = [], []
    for item in visible:
        start, stop = cache.nearest(item['start_ns']), cache.nearest(item['end_ns'],edge='end')
        display.append(item | {'start': cache.seconds(item['start_ns']), 'end': cache.seconds(item['end_ns'])})
        if stop > start:
            seeds.append({'id': item['id'], 'label': 'nonwear', 'status': 'pending',
                          'start_sample': start, 'stop_sample': stop, 'notes': ''})
    return {'identity': identity, 'sha256': diary.sha256, 'path': str(diary.path),
            'intervals': display, 'issues': issues, 'seeds': seeds}
