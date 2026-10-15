# SPDX-License-Identifier: Apache-2.0
"""Gamma 更新过程上的倒数估计器偏差实验。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt

from config import (EXP1, DT, SEED, R0, PHI, Q_SCALE, ALPHA_G, EPS,
                    P_INIT, COLORS, FIGSIZE_SINGLE, DPI, FIGURE_DIR)
from data_gen import generate_gamma_renewal, bin_events
from baselines import reciprocal_mean_bias
from observer import AdaptiveObserver


def run(save=True):
    """执行 Experiment I 并生成图表。"""
    rng = np.random.default_rng(SEED)

    true_lam = EXP1['true_lambda']
    kappas = EXP1['kappa_range']
    n_rep = EXP1['n_replications']
    n_events = EXP1['n_events_per_trial']

    # 存储结果
    bias_reciprocal = np.full((len(kappas), n_rep), np.nan)
    bias_observer = np.full((len(kappas), n_rep), np.nan)

    # 理论偏差比曲线 B(κ) = κ/(κ−1)
    kappa_theory = np.linspace(1.01, 5.0, 200)
    B_theory = kappa_theory / (kappa_theory - 1.0)

    print("[Exp I] 扫描形状参数 κ ...")
    for i, kappa in enumerate(kappas):
        for j in range(n_rep):
            # 生成 Gamma 更新过程
            events = generate_gamma_renewal(kappa, true_lam, n_events, rng)
            duration = events[-1] + 1.0   # 仿真时长定为最后事件 + 1秒

            # --- 倒数估计器偏差比 ---
            if kappa > 1.0:
                recip_mean = reciprocal_mean_bias(events)
                bias_reciprocal[i, j] = recip_mean / true_lam
            # κ ≤ 1 时 E[1/δt] 不存在, 保持 NaN

            # --- Observer 偏差比 ---
            counts, _ = bin_events(events, DT, duration)
            obs = AdaptiveObserver(DT, R0, PHI, Q_SCALE, ALPHA_G, EPS, P_INIT,
                                   enable_gating=False)
            estimates = obs.process_sequence(counts)
            # 跳过前 10% 的暖机期
            warmup = int(0.1 * len(estimates))
            obs_mean = np.mean(estimates[warmup:])
            bias_observer[i, j] = obs_mean / true_lam

        if (i + 1) % 5 == 0:
            print(f"  κ = {kappa:.2f} ({i+1}/{len(kappas)}) 完成")

    # 计算中位数和四分位数
    # 对 κ ≤ 1 行全为 NaN (倒数估计理论不存在), 抑制预期的 All-NaN 警告
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice')
            recip_median = np.nanmean(bias_reciprocal, axis=1)
            recip_q25 = np.nanpercentile(bias_reciprocal, 25, axis=1)
            recip_q75 = np.nanpercentile(bias_reciprocal, 75, axis=1)

            obs_median = np.nanmedian(bias_observer, axis=1)
            obs_q25 = np.nanpercentile(bias_observer, 25, axis=1)
            obs_q75 = np.nanpercentile(bias_observer, 75, axis=1)

    # ----------------------------------------------------------------
    # 绘图
    # ----------------------------------------------------------------
    _setup_mpl()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # 理论偏差比曲线
    ax.plot(kappa_theory, B_theory, 'k--', lw=1.2, label=r'Theory $\kappa/(\kappa-1)$')

    # 倒数估计器 (仅 κ > 1 的数据点)
    valid = ~np.isnan(recip_median)
    ax.plot(kappas[valid], recip_median[valid], 'o-',
            color=COLORS['reciprocal'], ms=4, lw=1.5,
            label='Reciprocal $1/\\delta t$')

    # κ ≤ 1 区: 标记为未定义
    undefined_mask = kappas <= 1.0
    if np.any(undefined_mask):
        ax.axvspan(kappas[undefined_mask].min() - 0.05,
                   1.0,
                   color='#FFCDD2', alpha=0.4, label=r'$\mathbb{E}[1/\delta t]$ undefined')

    # Observer
    ax.plot(kappas, obs_median, 's-',
            color=COLORS['proposed'], ms=4, lw=1.5,
            label='Proposed Observer')
    ax.fill_between(kappas, obs_q25, obs_q75,
                    color=COLORS['proposed'], alpha=0.15)

    # 无偏基线
    ax.axhline(1.0, color='gray', ls=':', lw=1.0, label=r'Unbiased $\mathcal{B}=1$')

    ax.set_xlabel(r'Shape parameter $\kappa$')
    ax.set_ylabel(r'Empirical bias ratio $\hat{\mathcal{B}}(\kappa)$')
    ax.set_ylim(0.5, 8.0)
    ax.set_xlim(0.4, 5.5)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title('Experiment I: Reciprocal Divergence Validation')

    fig.tight_layout()

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURE_DIR / 'reciprocal_divergence.pdf'
        fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
        print(f"[Exp I] 图表已保存: {out_path}")

    plt.close(fig)

    return dict(kappas=kappas,
                bias_reciprocal_median=recip_median,
                bias_observer_median=obs_median)


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
    print("[Exp I] 完成。")
