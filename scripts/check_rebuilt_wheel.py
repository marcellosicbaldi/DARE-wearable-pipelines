#!/usr/bin/env python3
"""Check that rebuilding the sdist preserves wheel file contents (ignoring ZIP timestamps)."""
import argparse
from pathlib import Path
from zipfile import ZipFile


def contents(directory: Path) -> dict[str, bytes]:
    wheels = list(directory.glob('*.whl'))
    if len(wheels) != 1:
        raise ValueError(f'Expected one wheel in {directory}')
    with ZipFile(wheels[0]) as archive:
        return {name: archive.read(name) for name in archive.namelist() if not name.endswith('/')}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('rebuilt', type=Path)
    args = parser.parse_args()
    before, after = contents(args.original), contents(args.rebuilt)
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    if changed:
        raise SystemExit('Rebuilt wheel differs: ' + ', '.join(changed))
    print('Source archive rebuilt with identical wheel file contents.')


if __name__ == '__main__':
    main()
