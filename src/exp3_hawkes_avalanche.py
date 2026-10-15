# SPDX-License-Identifier: Apache-2.0
"""Hawkes 自激事件过程上的强度跟踪实验。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from config import (PROFILE_B, DT, SEED, R0, PHI, Q_SCALE, ALPHA_G, EPS,
                    P_INIT, COLORS, FIGSIZE_SINGLE, DPI, FIGURE_DIR)
from data_gen import generate_hawkes, bin_events, hawkes_intensity
from observer import AdaptiveObserver
from baselines import make_standard_kf
from metrics import rmse, mise, cross_correlation_lag


def run(save=True):
    """执行 Experiment III 并生成图表。"""
    rng = np.random.default_rng(SEED)
    params = PROFILE_B
    duration = params['duration']
    mu = params['mu']
    alpha = params['alpha']
    beta = params['beta']

    # ---- 1. 生成 Hawkes 过程事件 ----
    events = generate_hawkes(mu, alpha, beta, duration, rng)
    counts, t_bins = bin_events(events, DT, duration)

    # Hawkes 条件强度 (ground truth, 依赖已实现事件)
    lam_true = hawkes_intensity(t_bins, events, mu, alpha, beta)

    print(f"[Exp III] Profile B (Hawkes): {len(events)} 个事件, "
          f"平均强度 = {len(events)/duration:.1f} events/s")

    # ---- 2. 提出方法 ----
    obs = AdaptiveObserver(DT, R0, PHI, Q_SCALE, ALPHA_G, EPS, P_INIT)
    lam_obs = obs.process_sequence(counts)
    K_gain_obs = np.array(obs.history['K_gain'])

    # ---- 3. 标准 KF (常数噪声, 用平均强度调参) ----
    # 平稳均值: μ / (1 − α)
    lam_avg = mu / (1.0 - alpha)
    skf = make_standard_kf(DT, lam_avg, R0, PHI, Q_SCALE, P_INIT)
    lam_skf = skf.process_sequence(counts)

    # ---- 4. 量化指标 ----
    methods = {
        'Proposed Observer': lam_obs,
        'Standard KF':       lam_skf,
    }
    print("\n[Exp III] ---- Profile B 量化结果 ----")
    print(f"{'Method':<22s} {'RMSE':>8s} {'MISE':>10s} {'Lag l* (s)':>10s}")
    print("-" * 52)
    results = {}
    for name, lam_est in methods.items():
        r = rmse(lam_true, lam_est)
        m = mise(lam_true, lam_est, DT)
        lag = cross_correlation_lag(lam_true, lam_est, DT)
        results[name] = dict(rmse=r, mise=m, lag=lag)
        print(f"{name:<22s} {r:8.3f} {m:10.2f} {lag:10.3f}")

    # ---- 5. 找到最大雪崩区间 (用于 inset) ----
    burst_idx = np.argmax(lam_true)
    window_half = int(10.0 / DT)   # ±10 秒窗口
    inset_lo = max(0, burst_idx - window_half)
    inset_hi = min(len(t_bins), burst_idx + window_half)

    # ---- 5. 绘图 (双面板: 跟踪 + Kalman 增益) ----
    _setup_mpl()
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6.5),
                                         gridspec_kw={'height_ratios': [3, 1.2]},
                                         sharex=True)

    # 上面板: 强度跟踪
    ax_top.plot(t_bins, lam_true, color=COLORS['ground_truth'],
                lw=1.5, label='Ground truth (conditional)', zorder=5)
    ax_top.plot(t_bins, lam_obs, color=COLORS['proposed'],
                lw=1.3, label='Proposed Observer', zorder=4)
    ax_top.plot(t_bins, lam_skf, color=COLORS['standard_kf'],
                lw=1.3, ls='--', label='Standard KF (const. $Q,R$)', zorder=3)

    ax_top.set_ylabel(r'Intensity $\hat{\lambda}(t)$ [events/s]')
    ax_top.legend(fontsize=8, loc='upper right')
    ax_top.set_title('Experiment III: Tracking Hawkes Avalanches (Profile B)')

    # 下面板: Kalman 增益 (全时域)
    ax_bot.plot(t_bins, K_gain_obs, color=COLORS['proposed'], lw=0.8,
                label='Kalman gain $K$')
    ax_bot.set_xlabel('Time [s]')
    ax_bot.set_ylabel('Kalman gain $K$')
    ax_bot.legend(fontsize=8, loc='upper right')

    fig.align_ylabels([ax_top, ax_bot])
    fig.tight_layout()

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURE_DIR / 'hawkes_tracking.pdf'
        fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
        print(f"\n[Exp III] 图表已保存: {out_path}")

    plt.close(fig)
    return results


def _setup_mpl():
    """配置 matplotlib 为期刊级风格。"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'mathtext.fontset': 'stix',
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })


if __name__ == '__main__':
    results = run(save=True)
    print("[Exp III] 完成。")
