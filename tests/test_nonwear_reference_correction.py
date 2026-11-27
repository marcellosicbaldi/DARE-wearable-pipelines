"""Private reference corrections must be explicit and scoped to one recording."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.rescore_nonwear_benchmark import load_correction


class ReferenceCorrectionTests(unittest.TestCase):
    def correction(self):
        return dict(subject='900001', visit='T0', annotation_id='reviewed_missing_episode',
                    requested_start='2024-01-01T08:00:00', requested_end='2024-01-01T08:30:00',
                    reason='Synthetic reviewed removal', authority='Synthetic reviewer')

    def load(self, value):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'correction.local.json'
            path.write_text(json.dumps(value))
            return load_correction(path)

    def test_private_correction_preserves_explicit_authority_and_boundaries(self):
        expected = self.correction()
        result = self.load(expected)
        for field, value in expected.items():
            self.assertEqual(result[field], value)
        self.assertEqual(result['algorithm_changes'], 'none; rescore existing predictions only')

    def test_incomplete_unsafe_or_nonchronological_corrections_are_rejected(self):
        invalid = [dict(self.correction(), subject='../elsewhere'),
                   dict(self.correction(), visit='../T0'),
                   dict(self.correction(), authority=''),
                   dict(self.correction(), requested_start='2024-01-02'),
                   dict(self.correction(), requested_start='2024-01-01T08:00:00Z'),
                   dict(self.correction(), extra='unknown')]
        missing = self.correction()
        missing.pop('authority')
        invalid.append(missing)
        invalid.extend([None, [], ['unexpected']])
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(value)
