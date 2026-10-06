"""Regressions for installed configuration lookup and distribution privacy checks."""
from contextlib import chdir
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
with patch.object(sys, 'path', [str(ROOT / 'scripts'), *sys.path]):
    spec = importlib.util.spec_from_file_location('distribution_check', ROOT / 'scripts/check_distributions.py')
    distribution_check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(distribution_check)


class PackagingTests(unittest.TestCase):
    def test_configuration_paths_follow_invocation_directory(self):
        from gp_pipeline.config import EmpaticaSleepConfig
        from ravenna_pipeline.config import RavennaSleepConfig
        from dare_wearables.common.paths import resolve_path
        with tempfile.TemporaryDirectory() as tmp, chdir(tmp):
            root = Path(tmp).resolve()
            (root / 'configs').mkdir()
            # Private paths remain synthetic and are never committed.
            for example, config in [('empatica_sleep', EmpaticaSleepConfig), ('ravenna_sleep', RavennaSleepConfig)]:
                text = (ROOT / 'configs' / (example + '.example.toml')).read_text()
                local = Path('configs') / (example + '.local.toml')
                local.write_text(text)
                config.from_toml(local)
            self.assertEqual(resolve_path('configs/test.toml'), root / 'configs/test.toml')
            self.assertEqual(resolve_path('sensor', base=root / 'configs'), root / 'configs/sensor')

    def test_archive_audit_rejects_private_data_and_unsafe_paths(self):
        for name, message in [('outputs/result.csv', 'private/generated directory'),
                              ('configs/run.local.toml', 'private configuration'),
                              ('synthetic.dist-info/study.csv', 'file type is not in the publication allowlist'),
                              ('../escaped.py', 'unsafe archive path')]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                wheel = Path(tmp) / 'synthetic.whl'
                with ZipFile(wheel, 'w') as archive:
                    archive.writestr(name, 'synthetic')
                issues = distribution_check.check_archive(wheel, expected_version='0.1.0')
                self.assertTrue(any(message in issue for issue in issues), issues)

    def test_archive_audit_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / 'synthetic.whl'
            link = ZipInfo('linked.py')
            link.create_system = 3
            link.external_attr = 0o120777 << 16
            with ZipFile(wheel, 'w') as archive:
                archive.writestr(link, 'target.py')
            self.assertIn('symlink in wheel', distribution_check.check_archive(wheel, expected_version='0.1.0'))


if __name__ == '__main__':
    unittest.main()
