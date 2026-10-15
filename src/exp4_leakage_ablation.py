# SPDX-License-Identifier: Apache-2.0
"""间歇事件过程上的休眠门控消融实验。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from config import (PROFILE_C, DT, SEED, R0, PHI, Q_SCALE, ALPHA_G, EPS,
                    P_INIT, TAU_SLOW, COLORS, FIGSIZE_SINGLE, DPI, FIGURE_DIR)
from data_gen import (profile_c_intensity, generate_nhpp, bin_events,
                      compute_lambda_max)
from observer import AdaptiveObserver
from baselines import EWMAEstimator
from metrics import normalized_leakage_mass, wake_up_lag


def run(save=True):
    """执行 Experiment IV 并生成图表与泄漏指标。"""
    rng = np.random.default_rng(SEED)
    params = PROFILE_C
    duration = params['duration']

    # ---- 1. 生成 Profile C 事件 ----
    lam_max = compute_lambda_max(profile_c_intensity, params, duration)
    events = generate_nhpp(profile_c_intensity, lam_max, duration, rng, params)
    counts, t_bins = bin_events(events, DT, duration)
    lam_true = profile_c_intensity(t_bins, params)

    print(f"[Exp IV] Profile C: {len(events)} 个事件, {len(t_bins)} 个 bin")

    # 构建结构零区掩码
    zero_mask = lam_true == 0.0

    # ---- 2. 三个消融变体 ----
    # (a) EWMA (慢速, 代表典型 IIR 滤波器)
    ewma = EWMAEstimator(TAU_SLOW, DT)
    lam_ewma = ewma.process_sequence(counts)

    # (b) Observer 无门控
    obs_no_gate = AdaptiveObserver(DT, R0, PHI, Q_SCALE, ALPHA_G, EPS, P_INIT,
                                   enable_gating=False)
    lam_no_gate = obs_no_gate.process_sequence(counts)

    # (c) Observer 有门控 (完整算法)
    obs_gated = AdaptiveObserver(DT, R0, PHI, Q_SCALE, ALPHA_G, EPS, P_INIT,
                                 enable_gating=True)
    lam_gated = obs_gated.process_sequence(counts)

    # ---- 3. 计算泄漏质量与唤醒滞后 ----
    variants = {
        'EWMA':                     lam_ewma,
        'Proposed (no gating)':     lam_no_gate,
        'Proposed (with gating)':   lam_gated,
    }

    # 找到第一个唤醒点 (第一个零区结束后的首个活跃 bin)
    # Profile C 的第二个活跃窗口从 t=40s 开始
    wake_time = params['active_windows'][1][0]   # 40.0 秒
    wake_idx = int(round(wake_time / DT))

    print("\n[Exp IV] ---- Profile C 泄漏消融结果 (tab:leakage_results) ----")
    print(f"{'Method':<28s} {'M̄_leak (ev/s)':>14s} {'Wake-up lag (bins)':>18s}")
    print("-" * 62)

    results = {}
    for name, lam_est in variants.items():
        leak = normalized_leakage_mass(lam_est, zero_mask, DT)
        w_lag = wake_up_lag(lam_est, lam_true, wake_idx, DT)
        results[name] = dict(leakage=leak, wake_lag=w_lag)
        print(f"{name:<28s} {leak:14.6f} {w_lag:18d}")

    # ---- 4. 绘图 (双面板: 全时域 + 唤醒瞬态) ----
    _setup_mpl()
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6.5),
                                         gridspec_kw={'height_ratios': [2, 1.2]})

    # ---- 上面板: 全时域跟踪 ----
    # 结构零区背景
    for t_start, t_end in _get_dormant_windows(params):
        ax_top.axvspan(t_start, t_end, color='#ECEFF1', alpha=0.6, zorder=0)

    ax_top.plot(t_bins, lam_true, color=COLORS['ground_truth'],
                lw=1.8, label='Ground truth', zorder=5)
    ax_top.plot(t_bins, lam_ewma, color=COLORS['ewma_slow'],
                lw=1.2, alpha=0.8, label=f'EWMA ($\\tau$={TAU_SLOW}s)')
    ax_top.plot(t_bins, lam_no_gate, color=COLORS['no_gating'],
                lw=1.2, ls='--', alpha=0.8, label='Proposed (no gating)')
    ax_top.plot(t_bins, lam_gated, color=COLORS['proposed'],
                lw=1.3, label='Proposed (with gating)', zorder=4)

    ax_top.set_xlabel('Time [s]')
    ax_top.set_ylabel(r'Intensity $\hat{\lambda}(t)$ [events/s]')
    ax_top.legend(fontsize=8, loc='upper right')
    ax_top.set_title('Experiment IV: Dormant-State Leakage Ablation (Profile C)')

    # ---- 下面板: 唤醒瞬态 (t=38~50s) ----
    inset_lo_t = wake_time - 2.0
    inset_hi_t = wake_time + 10.0
    inset_lo = max(0, int(round(inset_lo_t / DT)))
    inset_hi = min(len(t_bins), int(round(inset_hi_t / DT)))

    # 零区背景
    for t_start, t_end in _get_dormant_windows(params):
        if t_end > inset_lo_t and t_start < inset_hi_t:
            ax_bot.axvspan(max(t_start, inset_lo_t), min(t_end, inset_hi_t),
                           color='#ECEFF1', alpha=0.6, zorder=0)

    ax_bot.plot(t_bins[inset_lo:inset_hi], lam_true[inset_lo:inset_hi],
                color=COLORS['ground_truth'], lw=1.8, zorder=5)
    ax_bot.plot(t_bins[inset_lo:inset_hi], lam_ewma[inset_lo:inset_hi],
                color=COLORS['ewma_slow'], lw=1.2, alpha=0.8)
    ax_bot.plot(t_bins[inset_lo:inset_hi], lam_no_gate[inset_lo:inset_hi],
                color=COLORS['no_gating'], lw=1.2, ls='--', alpha=0.8)
    ax_bot.plot(t_bins[inset_lo:inset_hi], lam_gated[inset_lo:inset_hi],
                color=COLORS['proposed'], lw=1.3, zorder=4)

    ax_bot.axvline(wake_time, color='gray', ls=':', lw=0.8, label='Wake-up point')
    ax_bot.set_xlabel('Time [s]')
    ax_bot.set_ylabel(r'$\hat{\lambda}(t)$ [events/s]')
    ax_bot.set_xlim(inset_lo_t, inset_hi_t)
    ax_bot.set_title('Wake-up transient detail ($t = 38$--$50$ s)', fontsize=10)
    ax_bot.legend(fontsize=8, loc='upper right')

    fig.align_ylabels([ax_top, ax_bot])
    fig.tight_layout()

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURE_DIR / 'leakage_ablation.pdf'
        fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
        print(f"\n[Exp IV] 图表已保存: {out_path}")

    plt.close(fig)
    return results


def _get_dormant_windows(params):
    """从活跃窗口列表推导休眠窗口列表。"""
    windows = sorted(params['active_windows'])
    dormant = []
    duration = params['duration']
    if windows[0][0] > 0:
        dormant.append((0, windows[0][0]))
    for i in range(len(windows) - 1):
        dormant.append((windows[i][1], windows[i + 1][0]))
    if windows[-1][1] < duration:
        dormant.append((windows[-1][1], duration))
    return dormant


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
    print("[Exp IV] 完成。")
