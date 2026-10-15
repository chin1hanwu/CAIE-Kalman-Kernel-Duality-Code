# SPDX-License-Identifier: Apache-2.0
"""合成 Poisson 事件上的 Observer 与 Oracle KDE 运行时间基准。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
import numpy as np
import matplotlib.pyplot as plt

from config import (DT, R0, PHI, Q_SCALE, ALPHA_G, EPS, P_INIT,
                    COLORS, DPI, FIGURE_DIR)
from observer import AdaptiveObserver
from baselines import OracleAdaptiveKDE


# ====================================================================
# 配置
# ====================================================================

# 事件序列长度 (main.tex: N ∈ {10^3, 10^4, 10^5, 10^6})
N_VALUES = [1_000, 10_000, 100_000, 1_000_000]

# 每个长度重复次数 (main.tex: 20 次取中位数)
N_REPEATS = 20

# Oracle KDE 的最大事件数 (>10^5 会非常慢，设置上限)
ORACLE_MAX_N = 100_000

# 合成数据: 恒定强度 Poisson (仅测计算成本，不关心统计精度)
LAMBDA_CONST = 50.0  # events/s


# ====================================================================
# 基准测试核心
# ====================================================================

def generate_poisson_events(n_events, lam, rng):
    """生成恰好 n_events 个齐次 Poisson 事件。"""
    intervals = rng.exponential(1.0 / lam, size=n_events)
    return np.cumsum(intervals)


def benchmark_observer(events, dt):
    """测量 Observer 的单次运行壁钟时间。"""
    duration = events[-1] + dt
    n_bins = int(round(duration / dt))

    # 装箱
    counts = np.zeros(n_bins, dtype=int)
    bin_idx = np.floor(events / dt).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    for idx in bin_idx:
        counts[idx] += 1

    obs = AdaptiveObserver(dt, R0, PHI, Q_SCALE, ALPHA_G, EPS,
                           P_INIT, enable_gating=True)

    t0 = time.perf_counter()
    obs.process_sequence(counts)
    elapsed = time.perf_counter() - t0
    return elapsed


def benchmark_oracle_kde(events, dt):
    """测量 Oracle Adaptive KDE 的单次运行壁钟时间。"""
    duration = events[-1] + dt
    n_eval = int(round(duration / dt))
    t_eval = (np.arange(n_eval) + 0.5) * dt

    kde = OracleAdaptiveKDE(h_pilot=1.0)

    t0 = time.perf_counter()
    kde.estimate(events, t_eval)
    elapsed = time.perf_counter() - t0
    return elapsed


# ====================================================================
# 主实验
# ====================================================================

def run(save=True):
    """执行计算复杂度基准测试。"""
    print('=' * 60)
    print('  Section 7.4: Computational Profiling')
    print('=' * 60)

    rng = np.random.default_rng(42)

    observer_times = {}
    oracle_times = {}

    for N in N_VALUES:
        print('\nN = {:,d}'.format(N))

        obs_reps = []
        ora_reps = []

        for rep in range(N_REPEATS):
            events = generate_poisson_events(N, LAMBDA_CONST, rng)

            # Observer
            t_obs = benchmark_observer(events, DT)
            obs_reps.append(t_obs)

            # Oracle KDE (跳过太大的 N)
            if N <= ORACLE_MAX_N:
                t_ora = benchmark_oracle_kde(events, DT)
                ora_reps.append(t_ora)

        observer_times[N] = np.median(obs_reps)
        if ora_reps:
            oracle_times[N] = np.median(ora_reps)

        print('  Observer: median={:.4f} s'.format(observer_times[N]))
        if N in oracle_times:
            print('  Oracle KDE: median={:.4f} s'.format(oracle_times[N]))
        else:
            print('  Oracle KDE: skipped (N > {:,d})'.format(ORACLE_MAX_N))

    # ---- 汇总表 (tab:timing) ----
    print('\n[Profiling] ---- 壁钟时间汇总 (tab:timing) ----')
    header = '{:<20s}'.format('Method')
    for N in N_VALUES:
        header += ' {:>12s}'.format('N={:,d}'.format(N))
    print(header)
    print('-' * (20 + 13 * len(N_VALUES)))

    row_obs = '{:<20s}'.format('Proposed Observer')
    for N in N_VALUES:
        row_obs += ' {:12.4f}'.format(observer_times[N])
    print(row_obs)

    row_ora = '{:<20s}'.format('Oracle KDE')
    for N in N_VALUES:
        if N in oracle_times:
            row_ora += ' {:12.4f}'.format(oracle_times[N])
        else:
            row_ora += ' {:>12s}'.format('—')
    print(row_ora)

    # 计算每 bin 平均时间 (Observer 的循环粒度是 bin, 不是 event)
    print('\n每 bin 平均时间:')
    for N in N_VALUES:
        # n_bins ≈ (N / λ) / Δt + 1
        approx_duration = N / LAMBDA_CONST
        n_bins = int(round(approx_duration / DT)) + 1
        per_bin_us = observer_times[N] / n_bins * 1e6
        print('  N={:>9,d} ({} bins): {:.2f} µs/bin'.format(
            N, n_bins, per_bin_us))

    # ---- 绘制 log-log 图 (fig:complexity_scaling) ----
    if save:
        _plot_scaling(observer_times, oracle_times)

    return dict(observer=observer_times, oracle=oracle_times)


def _plot_scaling(observer_times, oracle_times):
    """绘制 log-log 复杂度图。"""
    _setup_mpl()

    fig, ax = plt.subplots(figsize=(7, 5))

    # Observer
    ns_obs = sorted(observer_times.keys())
    ts_obs = [observer_times[n] for n in ns_obs]
    ax.loglog(ns_obs, ts_obs, 'o-', color=COLORS['proposed'],
              lw=2, ms=8, label='Proposed Observer', zorder=5)

    # Oracle KDE
    ns_ora = sorted(oracle_times.keys())
    ts_ora = [oracle_times[n] for n in ns_ora]
    if ts_ora:
        ax.loglog(ns_ora, ts_ora, 's-', color='#E53935',
                  lw=2, ms=8, label='Oracle Adaptive KDE', zorder=4)

    # 参考线: O(N) 和 O(N²)
    n_ref = np.array([1e3, 1e7])
    # 以 observer 在 N=10^3 处为锚点
    anchor_t = observer_times[1000]
    t_linear = anchor_t * (n_ref / 1e3)
    t_quad = anchor_t * (n_ref / 1e3) ** 2
    ax.loglog(n_ref, t_linear, '--', color='gray', lw=1, alpha=0.5,
              label='$\\mathcal{O}(N)$ ref.')
    ax.loglog(n_ref, t_quad, ':', color='gray', lw=1, alpha=0.5,
              label='$\\mathcal{O}(N^2)$ ref.')

    ax.set_xlabel('Event count $N$')
    ax.set_ylabel('Wall-clock time [s]')
    ax.set_title('Computational Scaling')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', ls=':', alpha=0.3)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / 'complexity_scaling.pdf'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    print('\n[Profiling] 复杂度图已保存: {}'.format(out_path))
    plt.close(fig)


def _setup_mpl():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'mathtext.fontset': 'stix',
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })


if __name__ == '__main__':
    results = run(save=True)
    print('\n[Complexity] 基准测试完成。')
