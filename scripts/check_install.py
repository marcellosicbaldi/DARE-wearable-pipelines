#!/usr/bin/env python3
"""Exercise the installed distribution outside the checkout, including model loading."""
from __future__ import annotations

import argparse
from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heart-rate', action='store_true', help='Run real BeliefPPG inference on a synthetic signal')
    parser.add_argument('--expect-wheel', action='store_true', help='Reject imports from an editable source checkout')
    args = parser.parse_args()
    dist = metadata.distribution('DARE-wearable-pipelines')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
               TF_CPP_MIN_LOG_LEVEL='2', TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1')
    env.pop('PYTHONPATH', None)
    if args.expect_wheel:
        import dare_wearables
        location = Path(dare_wearables.__file__).resolve()
        if not location.is_relative_to(Path(dist.locate_file('')).resolve()):
            raise SystemExit('Package was imported from a source checkout instead of the installed wheel.')
    with tempfile.TemporaryDirectory() as tmp:
        for entry in dist.entry_points:
            if entry.group != 'console_scripts':
                continue
            script = f'import sys; from {entry.module} import {entry.attr}; sys.argv=[{entry.name!r}, "--help"]; {entry.attr}()'
            subprocess.run([sys.executable, '-c', script], cwd=tmp, env=env,
                           check=True, stdout=subprocess.DEVNULL, timeout=60)
        # Loading the bundled laterality model must not produce a version warning.
        script = '''
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.simplefilter('error', InconsistentVersionWarning)
from mobgap.laterality import LrcUllrich
LrcUllrich(**LrcUllrich.PredefinedParameters.msproject_all)
from dare_wearables.common.paths import resolve_path
from pathlib import Path
assert resolve_path('configs/example.toml') == Path.cwd() / 'configs/example.toml'
from gp_pipeline.config import EmpaticaSleepConfig
from ravenna_pipeline.config import RavennaSleepConfig
from dare_wearables.wrist.heart_rate_variability.pipeline import run_hrv_pipeline
'''
        subprocess.run([sys.executable, '-c', script], cwd=tmp, env=env, check=True, timeout=90)
        if args.heart_rate:
            script = '''
import numpy as np
from beliefppg import infer_hr_uncertainty
fs = 64
t = np.arange(120 * fs) / fs
rng = np.random.default_rng(42)
ppg = (np.sin(2 * np.pi * 1.2 * t) + 0.02 * rng.normal(size=t.size))[:, None]
acc = 0.01 * rng.normal(size=(t.size, 3))
hr, uncertainty, intervals = infer_hr_uncertainty(ppg, fs, acc=acc, acc_freq=fs, uncertainty='std')
assert hr.size > 0 and np.isfinite(hr).all()
assert np.isfinite(uncertainty).all() and len(hr) == len(uncertainty) == len(intervals)
print('Synthetic BeliefPPG inference passed:', hr.size, 'windows')
'''
            subprocess.run([sys.executable, '-c', script], cwd=tmp, env=env, check=True, timeout=180)
    print(f'Installed distribution {dist.version}: console commands, configuration paths and model loading passed.')


if __name__ == '__main__':
    main()
