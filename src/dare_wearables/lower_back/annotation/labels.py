"""Exact sample boundaries, explicit review, and independent atomic label storage."""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import tempfile
import threading

from .data import atomic_json, subject_id

FIELDS = ('subject', 'visit', 'annotation_id', 'label', 'part', 'start', 'end',
          'start_sample', 'stop_sample', 'duration_seconds', 'reviewer', 'notes',
          'diary_start', 'diary_end', 'raw_input_identity')
ROW_FIELDS = {'id', 'label', 'status', 'start_sample', 'stop_sample', 'notes'}


class RevisionConflict(ValueError):
    pass


class LabelStore:
    def __init__(self, root, subject, visit, cache, diary):
        self.subject, self.visit = subject_id(subject), visit
        self.cache, self.diary = cache, diary
        self.folder = Path(root) / self.subject / visit
        self.path = self.folder / 'annotations.json'
        self._lock = threading.RLock()

    def load(self):
        if not self.path.exists():
            return {'schema': 1, 'revision': 0, 'subject': self.subject, 'visit': self.visit,
                    'raw_input_identity': self.cache.meta['input_identity'], 'diary_identity': self.diary['identity'],
                    'reviewer': '', 'rows': self.diary['seeds']}
        record = json.loads(self.path.read_text())
        if (record.get('schema') != 1 or record.get('subject') != self.subject or record.get('visit') != self.visit
                or record.get('raw_input_identity') != self.cache.meta['input_identity']
                or record.get('diary_identity') != self.diary['identity']):
            raise ValueError('The recording or diary differs from the saved annotations. Use a separate output folder to review the changed source.')
        self.validate(record['rows'], record['reviewer'])
        return record

    def validate(self, rows, reviewer):
        if not isinstance(rows, list) or len(rows) > 5000:
            raise ValueError('Invalid annotation list.')
        if not isinstance(reviewer, str) or len(reviewer) > 120:
            raise ValueError('Reviewer name must be text, up to 120 characters.')
        ids, reviewed = set(), []
        source_ids = {item['id'] for item in self.diary['seeds']}
        for row in rows:
            if not isinstance(row, dict) or set(row) != ROW_FIELDS:
                raise ValueError('Annotation fields must be id, label, status, start_sample, stop_sample, notes.')
            ident = row['id']
            if not isinstance(ident, str) or not re.fullmatch(r'(?:diary_\d_\d|manual_[a-zA-Z0-9_-]{1,80})', ident):
                raise ValueError('Invalid annotation identifier.')
            if ident in ids or (ident.startswith('diary_') and ident not in source_ids):
                raise ValueError('Duplicate or unknown diary annotation.')
            ids.add(ident)
            if row['label'] not in ('nonwear', 'wear') or row['status'] not in ('pending', 'reviewed', 'rejected'):
                raise ValueError('Choose wear/nonwear and pending/reviewed/rejected.')
            a, b = row['start_sample'], row['stop_sample']
            if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(self.cache.times):
                raise ValueError('Boundaries must be ordered original-sample indices inside the recording.')
            if not isinstance(row['notes'], str) or len(row['notes']) > 4000:
                raise ValueError('Notes must be text, up to 4000 characters.')
            if row['status'] == 'reviewed':
                if not reviewer.strip():
                    raise ValueError('Enter your reviewer name before saving confirmed labels.')
                reviewed.append((a,b,ident))
        if not source_ids <= ids:
            raise ValueError('Keep original diary entries in the audit; reject them instead of deleting them.')
        reviewed.sort()
        if any(a < previous_b for (_,previous_b,_), (a,_,_) in zip(reviewed[:-1],reviewed[1:])):
            raise ValueError('Confirmed intervals overlap. Adjust or reject the overlapping annotation first.')

    @contextmanager
    def locked(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        with self._lock, (self.folder / '.labels.lock').open('a') as stream:
            # Coordinate browser tabs and independently launched local servers.
            try:
                import fcntl
            except ImportError:
                fcntl = None
            if fcntl:
                fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def save(self, rows, reviewer, expected_revision):
        if type(expected_revision) is not int:
            raise ValueError('Missing saved revision.')
        self.validate(rows, reviewer)
        with self.locked():
            previous = self.load()
            if previous['revision'] != expected_revision:
                raise RevisionConflict('A different tab saved newer changes. Reload this participant before saving again.')
            record = {'schema': 1, 'revision': expected_revision+1, 'subject': self.subject, 'visit': self.visit,
                      'reviewer': reviewer.strip(), 'saved_at_utc': datetime.now(timezone.utc).isoformat(),
                      'raw_input_identity': self.cache.meta['input_identity'],
                      'raw_source': self.cache.meta['source'], 'sample_rate_hz': self.cache.fs,
                      'timestamp_policy': self.cache.meta['timestamp_policy'],
                      'diary_identity': self.diary['identity'], 'diary_sha256': self.diary['sha256'],
                      'original_diary': self.diary['intervals'], 'diary_import_issues': self.diary['issues'],
                      'interval_contract': 'Original retained sample indices [start_sample, stop_sample); end timestamp exclusive.',
                      'coverage': 'Only reviewed intervals are labels; pending/rejected/unreviewed time is not inferred as wear.',
                      'rows': rows}
            old_rows = {row['id']:row for row in previous['rows']}
            record['reviewed_by'] = {}
            record['reviewed_at_utc'] = {}
            for row in rows:
                if row['status'] != 'reviewed':
                    continue
                before = old_rows.get(row['id'],{})
                unchanged = (before.get('status') == 'reviewed'
                             and all(before.get(key) == row[key] for key in ('label','start_sample','stop_sample')))
                record['reviewed_by'][row['id']] = (previous.get('reviewed_by',{}).get(row['id'],previous['reviewer'])
                                                   if unchanged else reviewer.strip())
                record['reviewed_at_utc'][row['id']] = (previous.get('reviewed_at_utc',{}).get(row['id'],previous.get('saved_at_utc'))
                                                       if unchanged else record['saved_at_utc'])
            if previous['revision']:
                atomic_json(self.folder / 'history' / f"revision_{previous['revision']:06d}.json", previous)
            # This single atomic file is authoritative. CSV is a regenerable convenience export.
            atomic_json(self.path, record)
            warning = None
            try:
                self.write_csv(record)
            except OSError as error:
                warning = f'Labels saved. CSV could not be written: {error}. Use Export CSV to regenerate it.'
            return record, warning

    def csv_text(self, record=None):
        record = self.load() if record is None else record
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        original = {item['id']: item for item in self.diary['intervals']}
        for row in sorted(record['rows'], key=lambda x: x['start_sample']):
            if row['status'] != 'reviewed':
                continue
            source = original.get(row['id'])
            part = 0
            # Missing time never becomes a wear/nonwear training example.
            for first, stop in self.cache.meta['segments']:
                a, b = max(first,row['start_sample']), min(stop,row['stop_sample'])
                if a >= b:
                    continue
                part += 1
                start_ns = self.cache.boundary_ns(a)
                end_ns = min(self.cache.boundary_ns(b), int(self.cache.times[b-1]) + round(1e9/self.cache.fs))
                from pandas import Timestamp
                writer.writerow({'subject': self.subject, 'visit': self.visit, 'annotation_id': row['id'],
                                 'label': row['label'], 'part': part, 'start': Timestamp(start_ns).isoformat(),
                                 'end': Timestamp(end_ns).isoformat(), 'start_sample': a, 'stop_sample': b,
                                 'duration_seconds': (b-a)/self.cache.fs,
                                 'reviewer': self.safe_text(record.get('reviewed_by',{}).get(row['id'],record['reviewer'])),
                                 'notes': self.safe_text(row['notes']),
                                 'diary_start': Timestamp(source['start_ns']).isoformat() if source else '',
                                 'diary_end': Timestamp(source['end_ns']).isoformat() if source else '',
                                 'raw_input_identity': self.cache.meta['input_identity']})
        return stream.getvalue()

    @staticmethod
    def safe_text(value):
        # Keep exported notes inert if a CSV is opened in a spreadsheet.
        return "'" + value if value.startswith(('=', '+', '-', '@', '\t', '\r')) else value

    def write_csv(self, record=None):
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / 'reviewed_intervals.csv'
        handle, name = tempfile.mkstemp(prefix='.csv-', dir=self.folder)
        try:
            with os.fdopen(handle, 'w', newline='') as stream:
                stream.write(self.csv_text(record))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
        return path

    def display_rows(self, record):
        return [row | {'start': self.cache.boundary(row['start_sample']),
                       'end': self.cache.boundary(row['stop_sample'],edge='end')} for row in record['rows']]
