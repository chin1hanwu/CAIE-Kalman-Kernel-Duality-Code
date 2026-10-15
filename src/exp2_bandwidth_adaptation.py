# SPDX-License-Identifier: Apache-2.0
"""标量与二维 Observer 的带宽缩放及非平稳跟踪实验。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import (PROFILE_A, DT, SEED, R0, PHI, Q_SCALE, ALPHA_G, EPS,
                    P_INIT, TAU_FAST, TAU_SLOW, COLORS, FIGSIZE_SINGLE,
                    FIGSIZE_DOUBLE, DPI, FIGURE_DIR)
from data_gen import (profile_a_intensity, generate_nhpp, bin_events,
                      compute_lambda_max)
from observer import AdaptiveObserver, AdaptiveObserver1D
from baselines import (EWMAEstimator, OracleAdaptiveKDE, make_standard_kf,
                        reciprocal_on_bins)
from metrics import rmse, mise, cross_correlation_lag


# ====================================================================
# Part 1: 缩放律数值验证 — 恒定 λ 下的稳态 τ_eff
# ====================================================================

def run_part1_scaling_law(save=True):
    """
    在多个恒定强度下运行 Observer，提取稳态 τ_eff，
    绘制 log-log 图验证 τ_eff ∝ λ^{-1/2} (1D) 和 τ_eff ∝ λ^{-1/4} (2D)。
    """
    print('\n[Part 1] 缩放律数值验证')
    print('-' * 60)

    rng = np.random.default_rng(SEED)

    # 扫描的恒定强度值 (覆盖 2 个数量级)
    lambda_values = np.array([1, 2, 5, 10, 20, 50, 100, 200])
    duration = 200.0     # 每个 λ 运行 200 秒确保达到稳态
    warmup_frac = 0.5    # 前 50% 为暖机期, 取后半段中位数

    tau_eff_1d = np.zeros(len(lambda_values))
    tau_eff_2d = np.zeros(len(lambda_values))
    K_gain_1d = np.zeros(len(lambda_values))
    K_gain_2d = np.zeros(len(lambda_values))

    for i, lam in enumerate(lambda_values):
        # 生成齐次 Poisson 过程
        n_expected = int(lam * duration * 1.5)
        iat = rng.exponential(1.0 / lam, size=n_expected)
        events = np.cumsum(iat)
        events = events[events < duration]

        counts, t_bins = bin_events(events, DT, duration)
        K = len(counts)
        warmup_k = int(K * warmup_frac)

        # 1D Observer
        obs_1d = AdaptiveObserver1D(DT, R0, PHI, Q_SCALE, ALPHA_G, EPS, P_INIT,
                                     enable_gating=False)
        obs_1d.process_sequence(counts)
        tau_1d_arr = np.array(obs_1d.history['tau_eff'][warmup_k:])
        K_1d_arr = np.array(obs_1d.history['K_gain'][warmup_k:])
        # 过滤有效值
        valid_1d = (tau_1d_arr > 0) & (tau_1d_arr < 1e6) & np.isfinite(tau_1d_arr)
        tau_eff_1d[i] = np.median(tau_1d_arr[valid_1d]) if np.any(valid_1d) else np.nan
        K_gain_1d[i] = np.median(K_1d_arr[valid_1d]) if np.any(valid_1d) else np.nan

        # 2D CV Observer
        obs_2d = AdaptiveObserver(DT, R0, PHI, Q_SCALE, ALPHA_G, EPS, P_INIT,
                                  enable_gating=False)
        obs_2d.process_sequence(counts)
        tau_2d_arr = np.array(obs_2d.history['tau_eff'][warmup_k:])
        K_2d_arr = np.array(obs_2d.history['K_gain'][warmup_k:])
        valid_2d = (tau_2d_arr > 0) & (tau_2d_arr < 1e6) & np.isfinite(tau_2d_arr)
        tau_eff_2d[i] = np.median(tau_2d_arr[valid_2d]) if np.any(valid_2d) else np.nan
        K_gain_2d[i] = np.median(K_2d_arr[valid_2d]) if np.any(valid_2d) else np.nan

        print('  λ={:>5.0f}:  τ_1D={:>7.3f}s (K={:.4f}),  τ_2D={:>7.3f}s (K={:.4f})'.format(
            lam, tau_eff_1d[i], K_gain_1d[i], tau_eff_2d[i], K_gain_2d[i]))

    # ---- 拟合 log-log 斜率 ----
    valid = np.isfinite(tau_eff_1d) & (tau_eff_1d > 0)
    if np.sum(valid) > 2:
        slope_1d, intercept_1d = np.polyfit(np.log10(lambda_values[valid]),
                                             np.log10(tau_eff_1d[valid]), 1)
        print('\n[Part 1] 1D Observer log-log 斜率: {:.3f} (理论值: -0.500)'.format(slope_1d))
    else:
        slope_1d = np.nan

    valid = np.isfinite(tau_eff_2d) & (tau_eff_2d > 0)
    if np.sum(valid) > 2:
        slope_2d, intercept_2d = np.polyfit(np.log10(lambda_values[valid]),
                                             np.log10(tau_eff_2d[valid]), 1)
        print('[Part 1] 2D CV Observer log-log 斜率: {:.3f} (理论值: -0.250)'.format(slope_2d))
    else:
        slope_2d = np.nan

    # ---- 理论参考曲线 ----
    lam_ref = np.logspace(np.log10(0.8), np.log10(250), 100)
    # 1D 理论: τ = sqrt(φΔt/(qλ)) = C / sqrt(λ)
    C_1d_theory = np.sqrt(PHI * DT / Q_SCALE)
    tau_theory_1d = C_1d_theory / np.sqrt(lam_ref)

    # 2D 理论: 用经验锚定 (理论斜率 -1/4)
    valid_2d = np.isfinite(tau_eff_2d) & (tau_eff_2d > 0)
    if np.any(valid_2d):
        anchor_idx = len(lambda_values) // 2
        C_2d = tau_eff_2d[anchor_idx] * lambda_values[anchor_idx] ** 0.25
    else:
        C_2d = 1.0
    tau_theory_2d = C_2d * lam_ref ** (-0.25)

    # ---- 绘制 log-log 图 ----
    if save:
        _plot_scaling_law(lambda_values, tau_eff_1d, tau_eff_2d,
                          lam_ref, tau_theory_1d, tau_theory_2d,
                          slope_1d, slope_2d)

    return dict(
        lambda_values=lambda_values,
        tau_eff_1d=tau_eff_1d, tau_eff_2d=tau_eff_2d,
        slope_1d=slope_1d, slope_2d=slope_2d,
    )


def _plot_scaling_law(lambda_values, tau_1d, tau_2d,
                       lam_ref, tau_theory_1d, tau_theory_2d,
                       slope_1d, slope_2d):
    """绘制 τ_eff vs λ 的 log-log 图。"""
    _setup_mpl()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # 理论参考线
    ax.loglog(lam_ref, tau_theory_1d, '--', color='#1565C0', lw=1.2, alpha=0.6,
              label=r'Theory $\tau \propto \lambda^{-1/2}$')
    ax.loglog(lam_ref, tau_theory_2d, ':', color='#7B1FA2', lw=1.2, alpha=0.6,
              label=r'CARE limit $\tau \propto \lambda^{-1/4}$')

    # 实测数据点
    valid_1d = np.isfinite(tau_1d) & (tau_1d > 0)
    valid_2d = np.isfinite(tau_2d) & (tau_2d > 0)

    ax.loglog(lambda_values[valid_1d], tau_1d[valid_1d], 'o-', color='#1565C0',
              lw=2, ms=8, label='1D Observer (slope={:.2f})'.format(slope_1d),
              zorder=5)
    ax.loglog(lambda_values[valid_2d], tau_2d[valid_2d], 's-', color='#7B1FA2',
              lw=2, ms=8, label='2D CV Observer (slope={:.2f})'.format(slope_2d),
              zorder=4)

    ax.set_xlabel(r'Constant intensity $\lambda$ [events/s]')
    ax.set_ylabel(r'Steady-state $\tau_{\mathrm{eff}}$ [s]')
    ax.set_title('Experiment II-A: Scaling Law Verification')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, which='both', ls=':', alpha=0.3)

    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / 'scaling_law_numerical.pdf'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    print('\n[Part 1] log-log 图已保存: {}'.format(out_path))
    plt.close(fig)


# ====================================================================
# Part 2: 非平稳跟踪展示 — Profile A 上的带宽自适应
# ====================================================================

def run_part2_tracking(save=True):
    """Profile A 上的跟踪对比与带宽自适应可视化。"""
    print('\n[Part 2] 非平稳跟踪展示 (Profile A)')
    print('-' * 60)

    rng = np.random.default_rng(SEED)
    params = PROFILE_A
    duration = params['duration']

    # ---- 1. 生成 Profile A 事件 ----
    lam_max = compute_lambda_max(profile_a_intensity, params, duration)
    events = generate_nhpp(profile_a_intensity, lam_max, duration, rng, params)
    counts, t_bins = bin_events(events, DT, duration)
    lam_true = profile_a_intensity(t_bins, params)

    print('[Part 2] Profile A: {} 个事件, {} 个 bin'.format(len(events), len(t_bins)))

    # ---- 2. 运行所有估计器 ----
    obs_2d = AdaptiveObserver(DT, R0, PHI, Q_SCALE, ALPHA_G, EPS, P_INIT)
    lam_obs = obs_2d.process_sequence(counts)
    tau_eff_2d = np.array(obs_2d.history['tau_eff'])

    obs_1d = AdaptiveObserver1D(DT, R0, PHI, Q_SCALE, ALPHA_G, EPS, P_INIT)
    lam_obs_1d = obs_1d.process_sequence(counts)
    tau_eff_1d = np.array(obs_1d.history['tau_eff'])

    ewma_fast = EWMAEstimator(TAU_FAST, DT)
    ewma_slow = EWMAEstimator(TAU_SLOW, DT)
    lam_fast = ewma_fast.process_sequence(counts)
    lam_slow = ewma_slow.process_sequence(counts)

    lam_avg = np.mean(lam_true)
    skf = make_standard_kf(DT, lam_avg, R0, PHI, Q_SCALE, P_INIT)
    lam_skf = skf.process_sequence(counts)

    lam_recip = reciprocal_on_bins(events, DT, duration)

    oracle = OracleAdaptiveKDE(h_pilot=1.0)
    lam_oracle = oracle.estimate(events, t_bins)

    # ---- 3. 计算指标 ----
    methods = {
        'Proposed (2D CV)':  lam_obs,
        'Proposed (1D)':     lam_obs_1d,
        'Oracle KDE':        lam_oracle,
        'EWMA (fast)':       lam_fast,
        'EWMA (slow)':       lam_slow,
        'Standard KF':       lam_skf,
        'Reciprocal':        lam_recip,
    }

    # 约束属性表
    constraints = {
        'Proposed (2D CV)':  ('Yes', 'O(1)',   'Yes'),
        'Proposed (1D)':     ('Yes', 'O(1)',   'Yes'),
        'Oracle KDE':        ('No',  'O(N^2)', 'Yes'),
        'EWMA (fast)':       ('Yes', 'O(1)',   'No'),
        'EWMA (slow)':       ('Yes', 'O(1)',   'No'),
        'Standard KF':       ('Yes', 'O(1)',   'No'),
        'Reciprocal':        ('Yes', 'O(1)',   'No'),
    }

    print('\n[Part 2] ---- Profile A 量化结果 (tab:profile_a_results) ----')
    print('{:<22s} {:>8s} {:>10s} {:>8s}  {:>6s} {:>6s} {:>6s}'.format(
        'Method', 'RMSE', 'MISE', 'Lag(s)', 'Caus.', 'Cplx.', 'Adapt.'))
    print('-' * 72)

    results = {}
    for name, lam_est in methods.items():
        r = rmse(lam_true, lam_est)
        m = mise(lam_true, lam_est, DT)
        lag = cross_correlation_lag(lam_true, lam_est, DT)
        c = constraints[name]
        results[name] = dict(rmse=r, mise=m, lag=lag)
        print('{:<22s} {:8.3f} {:10.2f} {:8.3f}  {:>6s} {:>6s} {:>6s}'.format(
            name, r, m, lag, c[0], c[1], c[2]))

    # ---- 4. 绘图: 双面板 ----
    if save:
        _plot_tracking(t_bins, lam_true, lam_obs, lam_obs_1d, lam_oracle,
                       lam_fast, lam_slow, tau_eff_1d, tau_eff_2d)

    return results


def _plot_tracking(t_bins, lam_true, lam_obs, lam_obs_1d, lam_oracle,
                    lam_fast, lam_slow, tau_eff_1d, tau_eff_2d):
    """双面板图: 强度跟踪 + 有效带宽。"""
    _setup_mpl()
    fig = plt.figure(figsize=FIGSIZE_DOUBLE)
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.08)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    # -- 上面板: 强度跟踪 --
    ax_top.plot(t_bins, lam_true, color=COLORS['ground_truth'],
                lw=1.8, label='Ground truth', zorder=5)
    ax_top.plot(t_bins, lam_obs, color=COLORS['proposed'],
                lw=1.3, label='Proposed (2D CV)', zorder=4)
    ax_top.plot(t_bins, lam_oracle, color=COLORS['oracle'],
                lw=1.3, ls='--', label='Oracle KDE', zorder=3)
    ax_top.plot(t_bins, lam_fast, color=COLORS['ewma_fast'],
                lw=1.0, alpha=0.7, label='EWMA fast ($\\tau$={}s)'.format(TAU_FAST))
    ax_top.plot(t_bins, lam_slow, color=COLORS['ewma_slow'],
                lw=1.0, alpha=0.7, label='EWMA slow ($\\tau$={}s)'.format(TAU_SLOW))
    ax_top.set_ylabel(r'Intensity $\hat{\lambda}(t)$ [events/s]')
    ax_top.legend(fontsize=8, ncol=2, loc='upper left')
    ax_top.set_title('Experiment II-B: Tracking on Profile A')
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # -- 下面板: 有效带宽 vs 理论法则 --
    lam_safe = np.maximum(profile_a_intensity(t_bins, PROFILE_A), EPS)

    # 1D 理论
    tau_theory_1d = np.sqrt(PHI * DT / (Q_SCALE * lam_safe))

    # 2D CARE 参考
    valid_2d = (tau_eff_2d > 0) & (tau_eff_2d < 1e6)
    if np.any(valid_2d):
        mid = len(lam_safe) // 2
        C_2d = np.median(tau_eff_2d[valid_2d]) * np.median(lam_safe[valid_2d]) ** 0.25
    else:
        C_2d = 1.0
    tau_theory_2d = C_2d * lam_safe ** (-0.25)

    # 裁剪 τ_eff 极端值 (初始瞬态 / 唤醒过渡)，仅影响可视化
    tau_plot_1d = np.clip(tau_eff_1d, 0.05, 50)
    tau_plot_2d = np.clip(tau_eff_2d, 0.05, 50)

    ax_bot.plot(t_bins, tau_plot_1d, color='#1565C0', lw=1.3, alpha=0.75,
                label=r'1D Observer $\tau_{\mathrm{eff}}$')
    ax_bot.plot(t_bins, tau_theory_1d, color='#1565C0', ls='--', lw=1.1,
                label=r'1D Theory ($\propto \lambda^{-1/2}$)')

    ax_bot.plot(t_bins, tau_plot_2d, color='#7B1FA2', lw=1.3, alpha=0.8,
                label=r'2D CV Observer $\tau_{\mathrm{eff}}$')
    ax_bot.plot(t_bins, tau_theory_2d, color='#7B1FA2', ls=':', lw=1.3,
                label=r'2D CARE limit ($\propto \lambda^{-1/4}$)')

    # 固定带宽 EWMA 参考线
    ax_bot.axhline(TAU_FAST, color=COLORS['ewma_fast'], ls='--', lw=0.6, alpha=0.4)
    ax_bot.axhline(TAU_SLOW, color=COLORS['ewma_slow'], ls='--', lw=0.6, alpha=0.4)

    ax_bot.set_xlabel('Time [s]')
    ax_bot.set_ylabel(r'$\tau_{\mathrm{eff}}$ [s]')
    ax_bot.set_yscale('log')
    ax_bot.set_ylim(0.05, 50)
    ax_bot.legend(fontsize=7, loc='upper right', ncol=2)

    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / 'bandwidth_adaptation.pdf'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    print('\n[Part 2] 图表已保存: {}'.format(out_path))
    plt.close(fig)


# ====================================================================
# 主入口
# ====================================================================

def run(save=True):
    """执行 Experiment II 全部任务。"""
    results_part1 = run_part1_scaling_law(save=save)
    results_part2 = run_part2_tracking(save=save)
    return dict(part1=results_part1, part2=results_part2)


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
    print('[Exp II] 完成。')
