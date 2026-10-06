#!/usr/bin/env python3
"""Audit wheel/sdist contents and write checksums for GitHub release assets."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
from pathlib import Path, PurePosixPath
import tarfile
import tomllib
import zipfile

from check_publication import check_content

PACKAGES = ('dare_wearables', 'gp_pipeline', 'ravenna_pipeline')


def check_archive(path: Path, *, expected_version: str) -> list[str]:
    issues = []
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            files = {info.filename: archive.read(info) for info in archive.infolist() if not info.is_dir()}
            if any((info.external_attr >> 16) & 0o170000 == 0o120000 for info in archive.infolist()):
                issues.append('symlink in wheel')
        prefix = ''
        required = [f'{package}/__init__.py' for package in PACKAGES]
        metadata_name = next((name for name in files if name.endswith('.dist-info/METADATA')), '')
    else:
        with tarfile.open(path, 'r:gz') as archive:
            files = {}
            for member in archive.getmembers():
                if member.isdir():
                    continue
                if not member.isfile():
                    issues.append('non-regular archive member')
                    continue
                files[member.name] = archive.extractfile(member).read()
        roots = {name.split('/')[0] for name in files}
        if len(roots) != 1:
            return ['source distribution must have one root']
        prefix = next(iter(roots)) + '/'
        required = [f'src/{package}/__init__.py' for package in PACKAGES]
        required += ['LICENSE', 'README.md', 'pyproject.toml', 'uv.lock', '.python-version',
                     'THIRD_PARTY_NOTICES.md', 'licenses/nimbaldetach-LICENSE.txt',
                     'licenses/neurokit2-LICENSE.txt', 'scripts/check_install.py',
                     'scripts/check_distributions.py', 'configs/empatica_sleep.example.toml']
        metadata_name = prefix + 'PKG-INFO'
    for name in required:
        if prefix + name not in files:
            issues.append(f'missing required file: {name}')
    for name, payload in files.items():
        relative = name.removeprefix(prefix)
        p = PurePosixPath(relative)
        if p.is_absolute() or '..' in p.parts:
            issues.append('unsafe archive path')
            continue
        generated = (p.parent.name.endswith('.dist-info') and p.name in {'METADATA', 'WHEEL', 'RECORD'}) or relative in {'PKG-INFO', 'setup.cfg'}
        generated = generated or (p.parent.name.endswith('.egg-info') and p.name == 'PKG-INFO')
        if generated:
            # Generated metadata uses suffix-free names; scan its text too.
            reasons = check_content('metadata.txt', payload)
        else:
            reasons = check_content(relative, payload)
        issues.extend(f'{relative}: {reason}' for reason in reasons)
    if metadata_name not in files:
        issues.append('missing distribution metadata')
    else:
        meta = BytesParser().parsebytes(files[metadata_name])
        if meta['Name'].lower().replace('_', '-') != 'dare-wearable-pipelines':
            issues.append('incorrect distribution name')
        if meta['License-Expression'] != 'MIT':
            issues.append('incorrect or missing project license expression')
        if meta['Version'] != expected_version:
            issues.append('incorrect distribution version')
        if 'Private :: Do Not Upload' not in meta.get_all('Classifier', []):
            issues.append('missing PyPI upload prevention classifier')
    if path.suffix == '.whl':
        for suffix in ('LICENSE', 'THIRD_PARTY_NOTICES.md', 'licenses/nimbaldetach-LICENSE.txt', 'licenses/neurokit2-LICENSE.txt'):
            if not any(name.endswith('/' + suffix) for name in files):
                issues.append(f'missing license or notice: {suffix}')
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', default=Path('dist'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / 'pyproject.toml').read_text())['project']['version']
    wheels = list(args.directory.glob('*.whl'))
    sources = list(args.directory.glob('*.tar.gz'))
    if len(wheels) != 1 or len(sources) != 1:
        raise SystemExit('Expected exactly one wheel and one source distribution in the output directory.')
    assets = wheels + sources
    issues = [f'{path.name}: {issue}' for path in assets for issue in check_archive(path, expected_version=version)]
    if issues:
        raise SystemExit('\n'.join(issues))
    (args.directory / 'SHA256SUMS.txt').write_text(''.join(
        f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n' for path in sorted(assets)))
    print('Distribution contents passed; SHA256SUMS.txt written.')


if __name__ == '__main__':
    main()
