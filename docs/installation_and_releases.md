# Installation and GitHub releases

## Supported environment

The release target is CPython 3.11, Linux x86_64 and Apple Silicon macOS. The
reference interpreter is 3.11.17 and the lockfile tool is uv 0.12.23. Windows,
Intel macOS and other Python versions are not currently supported release
profiles. The earlier unpinned environment used Python 3.12; the first pinned
profile uses 3.11 to accommodate BeliefPPG's TensorFlow Probability 0.22.1 stack.

The wheel contains platform-independent project code. Native dependencies and
trained models come from their own packages, so a `py3-none-any` wheel does not
imply that the full scientific environment works on every platform.

## Install a reviewed revision

Download the source archive from the desired GitHub release, or clone the
repository and check out its release tag. For version 0.2.0:

```bash
git clone https://github.com/marcellosicbaldi/DARE-wearable-pipelines.git
cd DARE-wearable-pipelines
git checkout v0.2.0
```

Install uv 0.12.23, then run:

```bash
uv sync --locked
```

For heart-rate inference:

```bash
uv sync --locked --extra heart-rate
```

The default installation includes sleep, circadian/activity, gait/posture,
HRV, clinical processing and aggregation. BeliefPPG is optional because its
TensorFlow stack is large. Use `uv run --no-sync COMMAND` after syncing, or
activate `.venv` and run commands directly. Avoid an implicit resync that drops
the selected extra. Add `--no-dev` to `uv sync` for analysis-only installations;
build and metadata-check tools are only needed by maintainers.

Copy the required `configs/*.example.toml` files to ignored `*.local.toml` files
and edit private paths. A relative `--config` path is relative to your shell's
current directory. DARE-FALLSPREDICT GP paths within the TOML must be absolute (or use `~`);
DARE-FALLSPREDICT retains its config-relative/input-root rules documented in its README.

The lockfile pins transitive dependencies and artifact hashes. It is not proof
of clinical equivalence, cross-platform numerical identity or validation on
study recordings. Record the Git commit, configuration and input provenance for
each analysis run. Do not save private provenance in the public repository.

## Install a wheel with pip

Use a fresh Python 3.11 environment and download the wheel plus `SHA256SUMS.txt`
from the same release. Verify its SHA-256 checksum before installing:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install ./dare_wearable_pipelines-0.2.0-py3-none-any.whl
```

For heart rate, use
`python -m pip install './dare_wearable_pipelines-0.2.0-py3-none-any.whl[heart-rate]'`.
Pip resolves transitive dependencies at installation time. For exact versions,
prefer the source release and `uv sync --locked`. Example configurations,
documentation and the lockfile are in the source archive; the wheel contains
runtime code and license notices. Never use `pip install DARE-wearable-pipelines`
without a local path: this project is not distributed through PyPI.

## Dependency decisions

- MobGap is retained at 1.0.0 and scikit-learn pinned to 1.6.1, matching the
  bundled model. The installation smoke test makes a model version warning fail.
- BeliefPPG 0.3.1 uses TensorFlow 2.15.1, tf-keras 2.15.1 and TensorFlow
  Probability 0.22.1. Synthetic inference loads the real packaged model on CPU.
  The upstream [BeliefPPG package](https://github.com/eth-siplab/BeliefPPG) and
  [TensorFlow Probability release](https://github.com/tensorflow/probability/releases/tag/v0.22.1)
  are the reference sources; the combination is checked by this project's smoke test.
- Scientific dependencies retain the versions used for the reviewed algorithms.
  Update their pins deliberately, regenerate `uv.lock` with `uv lock`, and run
  all checks. Do not refresh versions silently during an analysis campaign.
- `requirements.txt` delegates to the project metadata; it is not a second
  independently maintained dependency list.

## Build and review release assets

In a source checkout with the development environment installed:

```bash
uv sync --locked --extra heart-rate
uv pip check
uv run --no-sync python scripts/check_publication.py
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python scripts/check_install.py --heart-rate
uv build
uv run --no-sync python -m twine check --strict dist/*.whl dist/*.tar.gz
uv run --no-sync python scripts/check_distributions.py
uv build dist/*.tar.gz --wheel --out-dir rebuilt
uv run --no-sync python scripts/check_rebuilt_wheel.py dist rebuilt
```

Use empty `dist/` and `rebuilt/` directories when preparing a new version. The
archive checker requires exactly one wheel and one source archive and writes
`SHA256SUMS.txt`. It scans archive payloads as well as source files, including
metadata, private configurations, study data and unsafe archive members.

The GitHub Actions workflow repeats these checks on Linux and macOS, then
reinstalls the wheel and checks it outside the checkout. Successful Linux runs
provide a `github-release-candidate` artifact with the wheel, source archive and
checksums. These are build artifacts, not an automatically published release.

## Publish through GitHub

1. Review `LICENSE` and `THIRD_PARTY_NOTICES.md` for any newly incorporated
   code. The project uses MIT; preserve the project and upstream notices in
   wheel and source builds.
2. Set the version in `pyproject.toml`, run `uv lock`, update release notes and
   commit the reviewed publication tree. Do not import private repository history.
3. Push to the chosen GitHub repository and require the Linux and macOS checks
   to pass for the exact commit being released.
4. Create a matching `vVERSION` tag and a GitHub Release describing changes,
   supported platforms, installation steps and validation limits.
5. Attach the reviewed wheel, source archive and checksums from the successful
   CI run for that commit. Verify download and installation from those assets.

No PyPI credentials, trusted publishing configuration or upload command is
needed. The `Private :: Do Not Upload` metadata classifier intentionally blocks
PyPI uploads while preserving normal installation from GitHub assets.
