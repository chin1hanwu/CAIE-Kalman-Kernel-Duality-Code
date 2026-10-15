# SPDX-License-Identifier: Apache-2.0
"""覆盖在线估计、数值兼容性、数据接口和合成实验的回归测试。"""
import contextlib
import datetime
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from baselines import EWMAEstimator, OracleAdaptiveKDE, make_standard_kf
from config import DT, R0, PHI, Q_SCALE, ALPHA_G, EPS, P_INIT
from data_gen import bin_events, generate_hawkes, generate_nhpp
from metrics import mise, rmse
from metrics_realworld import (
    cumulative_predictive_loglik,
    predictive_loglikelihood_step,
    time_rescaling_intervals,
)
from observer import AdaptiveObserver, AdaptiveObserver1D


def observer(cls, **kwargs):
    return cls(DT, R0, PHI, Q_SCALE, ALPHA_G, EPS, P_INIT, **kwargs)


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.counts = np.random.default_rng(42).poisson(1.0, 200)

    def test_finite_nonnegative_and_recorded(self):
        for cls in (AdaptiveObserver1D, AdaptiveObserver):
            with self.subTest(cls=cls.__name__):
                model = observer(cls)
                output = model.process_sequence(self.counts)
                self.assertEqual(output.shape, self.counts.shape)
                self.assertTrue(np.all(np.isfinite(output)))
                self.assertTrue(np.all(output >= 0))
                self.assertTrue(all(len(v) == len(output) for v in model.history.values()))

    def test_reset_repeats_identically(self):
        for cls in (AdaptiveObserver1D, AdaptiveObserver):
            model = observer(cls)
            first = model.process_sequence(self.counts)
            np.testing.assert_array_equal(first, model.process_sequence(self.counts))

    def test_streaming_matches_sequence(self):
        for cls in (AdaptiveObserver1D, AdaptiveObserver):
            model = observer(cls)
            expected = model.process_sequence(self.counts)
            model.reset()
            actual = [model.step(k, count) for k, count in enumerate(self.counts, start=1)]
            np.testing.assert_allclose(actual, expected)

    def test_future_counts_do_not_change_prefix(self):
        changed = self.counts.copy()
        changed[100:] += 100
        for cls in (AdaptiveObserver1D, AdaptiveObserver):
            model = observer(cls)
            first = model.process_sequence(self.counts)
            second = model.process_sequence(changed)
            np.testing.assert_array_equal(first[:100], second[:100])

    def test_gate_dormancy_and_wakeup(self):
        counts = np.r_[np.full(20, 5), np.zeros(400), 5]
        for cls in (AdaptiveObserver1D, AdaptiveObserver):
            model = observer(cls)
            output = model.process_sequence(counts)
            self.assertTrue(any(model.history["dormant_flag"]))
            self.assertEqual(output[-2], 0)
            self.assertGreater(output[-1], 0)
            self.assertFalse(model.history["dormant_flag"][-1])

    def test_empty_counts(self):
        for cls in (AdaptiveObserver1D, AdaptiveObserver):
            self.assertEqual(len(observer(cls).process_sequence(np.array([]))), 0)

    def test_covariance_is_positive_semidefinite(self):
        model = observer(AdaptiveObserver, enable_gating=False)
        model.process_sequence(self.counts)
        np.testing.assert_allclose(model.P, model.P.T, atol=1e-10)
        self.assertGreaterEqual(np.linalg.eigvalsh(model.P).min(), -1e-10)


class BaselineAndMetricTests(unittest.TestCase):
    def test_baselines_finite(self):
        counts = np.random.default_rng(3).poisson(1.0, 100)
        for model in (EWMAEstimator(0.5, DT), make_standard_kf(DT, 10, R0, PHI, Q_SCALE, P_INIT)):
            result = model.process_sequence(counts)
            self.assertTrue(np.all(np.isfinite(result)))
            self.assertTrue(np.all(result >= 0))

    def test_kde_empty_and_small(self):
        model = OracleAdaptiveKDE()
        times = np.linspace(0, 2, 20)
        np.testing.assert_array_equal(model.estimate(np.array([]), times), np.zeros(20))
        self.assertTrue(np.all(np.isfinite(model.estimate(np.array([0.2, 0.5, 1.5]), times))))

    def test_error_metrics_known_values(self):
        actual = np.array([1., 3.])
        estimate = np.array([2., 4.])
        self.assertEqual(rmse(actual, estimate), 1)
        self.assertAlmostEqual(mise(actual, estimate, 0.1), 0.2)

    def test_poisson_loglik_known_values(self):
        self.assertAlmostEqual(predictive_loglikelihood_step(2., 0, 0.5), -1)
        self.assertAlmostEqual(predictive_loglikelihood_step(2., 2, 0.5), -1 - np.log(2))
        total, curve = cumulative_predictive_loglik(np.full(3, 2.), np.array([0, 1, 2]), 0.5)
        self.assertAlmostEqual(total, -3 - np.log(2))
        self.assertAlmostEqual(curve[-1], total)

    def test_time_rescaling_constant_rate(self):
        intervals = time_rescaling_intervals(np.array([0.2, 0.5, 0.9]), lambda t: np.full_like(t, 5.), DT)
        np.testing.assert_allclose(intervals, [1.5, 2.0], atol=1e-12)


class DataTests(unittest.TestCase):
    def test_bin_conservation(self):
        counts, centers = bin_events(np.array([0.01, 0.09, 0.11, 0.99]), 0.1, 1.)
        self.assertEqual(counts.sum(), 4)
        self.assertEqual(counts[0], 2)
        self.assertEqual(len(centers), 10)

    def test_nhpp_reproducibility(self):
        def constant(t, params):
            return np.full_like(t, 5.0)
        args = (constant, 5.0, 5.)
        first = generate_nhpp(*args, np.random.default_rng(7), {})
        second = generate_nhpp(*args, np.random.default_rng(7), {})
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(np.diff(first) > 0))
        self.assertTrue(np.all((first >= 0) & (first < 5)))

    def test_hawkes_bounded_sorted(self):
        events = generate_hawkes(2., 0.5, 5., 5., np.random.default_rng(8))
        self.assertTrue(np.all(np.diff(events) > 0))
        self.assertTrue(np.all((events >= 0) & (events < 5)))

    def test_finance_synthetic_csv(self):
        from app_finance import load_aggtrades, bin_trades
        output = ROOT / "outputs" / "test-artifacts"
        output.mkdir(parents=True, exist_ok=True)
        origin = int(datetime.datetime(2025, 10, 10, tzinfo=datetime.timezone.utc).timestamp() * 1_000_000)
        path = output / "synthetic-trades.csv"
        rows = [[0, 100, 2, 0, 0, origin + 1_500_000], [1, 101, 3, 0, 0, origin + 500_000]]
        np.savetxt(path, rows, delimiter=",", fmt="%.0f")
        with contextlib.redirect_stdout(io.StringIO()):
            times, prices, quantities = load_aggtrades([path])
        np.testing.assert_allclose(times, [0.5, 1.5])
        np.testing.assert_allclose(prices, [101, 100])
        np.testing.assert_allclose(quantities, [3, 2])
        counts, _ = bin_trades(times, 1., 2.)
        np.testing.assert_array_equal(counts, [1, 1])


class IntegrationTests(unittest.TestCase):
    def test_synthetic_experiments_render(self):
        import exp1_reciprocal_divergence as exp1
        import exp2_bandwidth_adaptation as exp2
        import exp3_hawkes_avalanche as exp3
        import exp4_leakage_ablation as exp4
        output = ROOT / "outputs" / "test-artifacts"
        output.mkdir(parents=True, exist_ok=True)
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(warnings.catch_warnings())
            warnings.filterwarnings("ignore", message=".*tight_layout.*")
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            for module in (exp1, exp2, exp3, exp4):
                stack.enter_context(patch.object(module, "FIGURE_DIR", output))
            stack.enter_context(patch.dict(exp1.EXP1, kappa_range=np.array([0.8, 1.2, 2.]), n_replications=2, n_events_per_trial=100))
            exp1.run(save=True)
            exp2.run(save=True)
            exp3.run(save=True)
            exp4.run(save=True)
        for name in ("reciprocal_divergence", "scaling_law_numerical", "bandwidth_adaptation", "hawkes_tracking", "leakage_ablation"):
            self.assertGreater((output / (name + ".pdf")).stat().st_size, 100)

    def test_complexity_small_input(self):
        from app_complexity import generate_poisson_events, benchmark_observer, benchmark_oracle_kde
        events = generate_poisson_events(50, 50., np.random.default_rng(4))
        self.assertGreaterEqual(benchmark_observer(events, DT), 0)
        self.assertGreaterEqual(benchmark_oracle_kde(events, DT), 0)

    @unittest.skipUnless(importlib.util.find_spec("pynwb"), "需要可选 pynwb 依赖")
    def test_neuro_synthetic_nwb(self):
        from pynwb import NWBHDF5IO, NWBFile
        from app_neuro import load_mc_maze, select_top_units, psth_estimate
        output = ROOT / "outputs" / "test-artifacts"
        output.mkdir(parents=True, exist_ok=True)
        path = output / "synthetic-neuro.nwb"
        nwb = NWBFile("合成接口测试", "synthetic", datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc))
        nwb.add_unit(id=0, spike_times=[0.1, 0.4, 1.0])
        nwb.add_unit(id=1, spike_times=[0.2, 0.6, 0.9, 1.1])
        with NWBHDF5IO(str(path), "w") as handle:
            handle.write(nwb)
        trains, unit_ids, duration = load_mc_maze(path)
        self.assertEqual(len(trains), 2)
        self.assertEqual(len(unit_ids), 2)
        self.assertGreaterEqual(duration, 1.1)
        self.assertEqual(select_top_units(trains, 1, duration), [1])
        rates, centers = psth_estimate(trains[1], 0.05, 5.0)
        self.assertEqual(rates.shape, centers.shape)
        self.assertTrue(np.all(np.isfinite(rates)))


if __name__ == "__main__":
    unittest.main()
