"""Independent numerical and scoring checks for the frozen research benchmark."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts import benchmark_lower_back_nonwear as bench
from dare_wearables.lower_back.nonwear import ggir2023
from dare_wearables.lower_back.annotation.data import write_raw_cache, atomic_json
from dare_wearables.common.output_state import _digest


class BenchmarkTests(unittest.TestCase):
    def test_candidates_are_fixed_unique_and_include_production(self):
        declared = bench.candidates()
        self.assertEqual(len(declared), 27)
        self.assertEqual(len({c.id for c in declared}), 27)
        self.assertIn(bench.BASELINE, {c.id for c in declared})
        self.assertEqual({c.minimum_minutes for c in declared}, {0, 5, 15})

    def test_fast_scores_match_direct_window_calculations_at_both_steps(self):
        fs = 5
        rng = np.random.default_rng(65)
        n = 7*3600*fs+83
        t = np.arange(n)/fs
        axes = []
        for a in range(3):
            x = rng.normal(0, .001, n) + .2*a
            moving = ((t//1100+a) % 4) == 0
            x[moving] += .05*np.sin(t[moving])
            axes.append(x)
        for window, step in ((60, 15), (30, 15), (30, 1), (15, 1)):
            with self.subTest(window=window, step=step):
                medium = step*60*fs
                count = n//medium
                # The upstream scoring function supports these window epochs
                # independently of its production wrapper's fixed 15-min grid.
                expected, _ = ggir2023._epoch_scores(axes, fs, medium, count, window_epochs=window//step)
                actual = bench.fast_scores(axes, fs, window, step)
                np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(bench.fast_scores([x[:10] for x in axes], fs, 60, 15)), 0)

    def test_fast_scores_strict_sd_and_range_boundaries(self):
        n = 4500
        sd_limit = .013*np.sqrt((n-1)/n)
        for amplitude in (np.nextafter(sd_limit, 0), sd_limit, np.nextafter(sd_limit, np.inf)):
            x = np.resize([-amplitude, amplitude], n)
            expected = ggir2023._epoch_scores([x]*3, 5., n, 1, window_epochs=1)[0]
            np.testing.assert_array_equal(bench.fast_scores([x]*3, 5., 15, 15), expected)
        for spread in (.0499999, .05, .0500001):
            x = np.zeros(n); x[-2:] = [-spread/2, spread/2]
            expected = ggir2023._epoch_scores([x]*3, 5., n, 1, window_epochs=1)[0]
            np.testing.assert_array_equal(bench.fast_scores([x]*3, 5., 15, 15), expected)

    def test_fifteen_minute_postprocessing_exactly_matches_production(self):
        rng = np.random.default_rng(32)
        for count in (0, 1, 2, 10, 12, 24, 50, 672):
            for _ in range(25):
                scores = np.repeat(rng.integers(0, 4, count), 5)[:count]
                expected = ggir2023._wear_decisions(scores, np.zeros(count), end_of_recording_correction=False)[2]
                np.testing.assert_array_equal(bench.decisions(scores, 2, 15), expected)

    def test_postprocessing_preserves_physical_hour_rules(self):
        # Two long nonwear bouts around one hour of wear: fill at either grid.
        scores = np.r_[np.zeros(16), np.full(16, 3), np.zeros(4), np.full(16, 3), np.zeros(16)]
        coarse = bench.decisions(scores, 2, 15)
        fine = bench.decisions(np.repeat(scores, 15), 2, 1)
        np.testing.assert_array_equal(np.repeat(coarse, 15), fine)
        # Three-axis eligibility is separate from postprocessing.
        np.testing.assert_array_equal(bench.decisions([2, 3, 2], 3, 15, False), [False, True, False])

    def test_interval_scoring_uses_exact_observed_overlap(self):
        truth = np.array([[10, 30], [60, 90]])
        prediction = np.array([[0, 20], [25, 70]])
        result = bench.confusion(prediction, truth, [[0, 100]], 1)
        self.assertEqual(result, dict(tp_seconds=25., fp_seconds=40., fn_seconds=25., tn_seconds=10.))
        stats = bench.rates(result)
        self.assertAlmostEqual(stats['recall'], .5)
        self.assertAlmostEqual(stats['precision'], 25/65)
        self.assertEqual(bench.length(bench.split_bounds([[10, 90]], [[0, 40], [40, 100]])), 80)
        np.testing.assert_array_equal(bench.mask_bouts([False, True, True, False], 10), [[10, 30]])

    def test_event_matching_is_one_to_one_and_counts_misses(self):
        class Clock:
            fs = 1
            def boundary_ns(self, s, edge='start'): return s*10**9
        truth = np.array([[0, 10], [20, 30], [50, 60]])
        pred = np.array([[0, 30], [50, 60]])
        summary, detail = bench.event_metrics(pred, truth, [{'id':str(i)} for i in range(3)], Clock())
        self.assertEqual(summary['matched_events_iou50'], 1)
        self.assertEqual(summary['any_overlap_events'], 3)
        self.assertTrue(detail[2]['matched_iou50'])
        self.assertEqual(detail[2]['start_error_seconds'], 0)

    def test_day_night_support_omits_unobserved_time(self):
        times = pd.date_range('2024-01-01 21:59:58', periods=6, freq='s').asi8
        support = bench.periods(times)
        self.assertEqual(bench.length(support['day']), 2)
        self.assertEqual(bench.length(support['night']), 4)
        self.assertEqual(bench.length(support['all']), 6)

    def test_gap_spanning_reference_keeps_one_episode_and_excludes_synthetic_samples(self):
        class Cache:
            fs = 100
            meta = {'segments': [[0, 100], [100, 200]]}
            def boundary_ns(self, s, edge='start'):
                if edge == 'end' and s == 100: return 10**9
                return s*10**7 + (260_000_000 if s >= 100 else 0)
        session = {'subject': '900001', 'rows': [{'id': 'manual_1', 'label': 'nonwear', 'status': 'reviewed',
                                               'start_sample': 50, 'stop_sample': 150}]}
        with TemporaryDirectory() as temp:
            rows, bounds = bench.reference(session, Cache(), Path(temp))
            exported = pd.read_csv(Path(temp)/'reference_episodes.csv')
            ledger = pd.read_csv(Path(temp)/'reference_gap_timestamps.csv')
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported.observed_parts.iloc[0], 2)
        self.assertEqual(exported.observed_seconds.iloc[0], 1.)
        self.assertEqual(len(ledger), 26)
        self.assertFalse(ledger.scorable.any())
        self.assertFalse(ledger.observed.any())

    def test_complete_subject_pipeline_on_invented_signals(self):
        fs = 5
        n = fs*3*3600
        times = pd.date_range('2024-01-01', periods=n, freq='200ms')
        moving = .12*np.sin(np.arange(n)/fs)
        moving[fs*3600:fs*5400] = 0
        raw = pd.DataFrame({'datetime': times,
            **{name: moving + i*.3 for i, name in enumerate(('ax', 'ay', 'az'))},
            **{name: np.zeros(n) for name in ('gx', 'gy', 'gz')}})
        with TemporaryDirectory() as temp:
            run = Path(temp)
            folder = run/'900001'/'T0'; folder.mkdir(parents=True)
            cache = run/'cache'
            write_raw_cache(cache, raw, source=run/'invented.OMX', identity='invented', fs=fs, hours=168, policy='strict')
            atomic_json(folder/'annotations.json', {'subject': '900001', 'rows': [
                {'id': 'manual_1', 'status': 'reviewed', 'label': 'nonwear',
                 'start_sample': fs*3600, 'stop_sample': fs*5400}]})
            (folder/'reviewed_intervals.csv').write_text('synthetic fixture\n')
            item = {'subject': '900001', 'visit': 'T0', 'cache': str(cache),
                'cache_metadata_sha256': _digest(cache/'metadata.json'),
                'labels_sha256': _digest(folder/'annotations.json'),
                'export_sha256': _digest(folder/'reviewed_intervals.csv'),
                'cache_files': {name: {'size': (cache/name).stat().st_size,
                                      'mtime_ns': (cache/name).stat().st_mtime_ns}
                                for name in ('signals.npy', 'timestamps.npy')}}
            bench.process_subject(item, run, bench.candidates())
            metrics = pd.read_csv(folder/'metrics.csv')
            events = pd.read_csv(folder/'events.csv')
            self.assertEqual(len(metrics), 81)
            self.assertEqual(len(events), 27)
            self.assertEqual(set(metrics.candidate), {c.id for c in bench.candidates()})
            self.assertTrue((metrics[['tp_seconds', 'fp_seconds', 'fn_seconds', 'tn_seconds']] >= 0).all().all())
            all_time = metrics.loc[metrics.period == 'all']
            np.testing.assert_allclose(all_time.tp_seconds+all_time.fn_seconds, 1800)
            np.testing.assert_allclose(all_time[['tp_seconds', 'fp_seconds', 'fn_seconds', 'tn_seconds']].sum(axis=1), 10800)
            self.assertTrue((folder/'verification.json').is_file())


if __name__ == '__main__':
    unittest.main()
