# Third-party notices and provenance

DARE-wearable-pipelines is licensed under the MIT License in `LICENSE`,
copyright (c) 2026 Marcello Sicbaldi. Incorporated third-party code retains its
original copyright and license notices below. Dependencies retain their own
licenses; the project license does not replace them.

## Incorporated source

- `src/dare_wearables/wrist/nonwear/nimbaldetach.py` derives from
  [nimbal/nimbaldetach](https://github.com/nimbal/nimbaldetach), by Adam Vert and
  Kit Beyer, with an embedded filter attribution to Kyle Weber. The upstream
  MIT notice is retained in `licenses/nimbaldetach-LICENSE.txt`. Source reviewed
  at upstream commit `275d267ea63efa890b8d70d682606902b28d1220`.
- `src/dare_wearables/wrist/heart_rate_variability/kubios.py` contains adapted
  NeuroKit peak-correction code. Preserve the MIT notice in
  `licenses/neurokit2-LICENSE.txt`; see
  [NeuroKit signal_fixpeaks](https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/signal/signal_fixpeaks.py).
  Local changes include accepting interval data and reporting artifact handling.
  The name Kubios refers to the published method, not bundled Kubios software.

- `src/dare_wearables/wrist/heart_rate_variability/ppg_beat_detection.py` contains
  the local MSPTDfast adaptation from NeuroKit2, as confirmed by the maintainer.
  The NeuroKit MIT notice is retained in `licenses/neurokit2-LICENSE.txt`.
  See [NeuroKit PPG peak detection](https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/ppg/ppg_findpeaks.py).
  The implementation also retains the Charlton et al. method citation in source.

The exact historical revisions used for these adaptations were not recorded in
the supplied source. The review date is 2026-10-06; the upstream commit above
identifies the inspected reference, not a claimed original import revision.

## Runtime dependencies and methods

Runtime packages are installed separately under their respective licenses.
`uv.lock` records the selected versions and artifact hashes. Model files supplied
by MobGap and BeliefPPG remain in those dependency distributions.

- [MobGap](https://github.com/mobilise-d/mobgap): Mobilise-D gait algorithms;
  Apache-2.0. The local gait orchestration retains its original attribution
  “Adapted by Jose AS on 09.06.2025”.
- [BeliefPPG](https://github.com/eth-siplab/BeliefPPG): Bieri, Streli, Demirel and
  Holz, *BeliefPPG: Uncertainty-aware Heart Rate Estimation from PPG signals via
  Belief Propagation*, UAI 2023; MIT.
- [NeuroKit2](https://github.com/neuropsychology/NeuroKit): physiological signal
  processing; MIT.
- DETACH: Vert et al. (2022), *Detecting accelerometer non-wear periods using
  change in acceleration combined with rate-of-change in temperature*,
  [doi:10.1186/s12874-022-01633-6](https://doi.org/10.1186/s12874-022-01633-6).
- MSPTDfast, van Hees sleep/nonwear methods, GGIR-inspired calibration, and the
  Lipponen–Tarvainen artifact method retain their method references in source.

## Maintainer-confirmed provenance

Marcello Sicbaldi authorized the project's MIT release on 2026-10-06 and
confirmed the following origins:

- The local MSPTDfast implementation was adapted from NeuroKit2 and is covered
  by the incorporated-source notice above.
- The GGIR-inspired routines were developed with inspiration from GGIR. They
  retain their method references and explanatory comments; GGIR is acknowledged
  as a methodological source. See [GGIR](https://github.com/wadpac/GGIR).
- Jose AS is a colleague who contributed the gait adaptation. The existing
  source credit is retained; the maintainer confirmed that no further action
  is needed for this contribution.

The historical upstream revision numbers for adaptations remain unrecorded.
This provenance note records the maintainer's clarification without inventing
source revisions or changing existing third-party notices.
