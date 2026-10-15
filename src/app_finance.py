# SPDX-License-Identifier: Apache-2.0
"""基于本地 Binance aggTrades CSV 的交易到达率分析。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import csv
import datetime
import matplotlib.pyplot as plt

from config import (DT, R0, PHI, Q_SCALE, ALPHA_G, EPS, P_INIT,
                    COLORS, FIGSIZE_SINGLE, DPI, FIGURE_DIR)
from observer import AdaptiveObserver1D
from baselines import EWMAEstimator, StandardKalmanFilter, make_standard_kf
from metrics_realworld import cumulative_predictive_loglik


# ====================================================================
# 配置
# ====================================================================

DATA_DIR = Path(__file__).parent.parent / 'data' / 'finance'
CSV_FILES = [
    DATA_DIR / 'BTCUSDT-aggTrades-2025-10-10.csv',
    DATA_DIR / 'BTCUSDT-aggTrades-2025-10-11.csv',
]

# 分析用 bin 宽度 (秒) — 金融环境用更小的 bin
FINANCE_DT = 1.0   # 1 秒 bin (金融数据密度远高于神经数据)

# 三体制时间窗口定义 (以 Oct 10 00:00 UTC 为 t=0)
# 基于 K 线数据分析:
#   - 事件核心: Oct 10 15:00-23:00 UTC (崩盘 + 极端波动)
#   - 恢复期: Oct 11 全天
REGIME_BOUNDARIES = {
    'pre_event': (0, 15 * 3600),          # Oct 10 00:00 ~ 15:00 UTC (15h)
    'event':     (15 * 3600, 24 * 3600),  # Oct 10 15:00 ~ 24:00 UTC (9h)
    'post_event': (24 * 3600, 48 * 3600), # Oct 11 全天 (24h)
}

# Observer 参数: 金融数据调整 (bin 宽度更小, 事件率更高)
FINANCE_R0 = 0.1
FINANCE_PHI = 1.0
FINANCE_Q_SCALE = 0.0003
FINANCE_ALPHA_G = 0.001

# EWMA 基线 (多带宽对比, 证明 10s 是 oracle-optimal 选择)
TAU_5 = 5.0        # 5 秒 (极快, 对 Poisson 噪声敏感)
TAU_FAST = 10.0    # 10 秒
TAU_15 = 15.0      # 15 秒
TAU_SLOW = 300.0   # 5 分钟 (慢速, 无法跟踪闪崩)


# ====================================================================
# 数据加载
# ====================================================================

def load_aggtrades(csv_paths, t_origin_date='2025-10-10'):
    """
    加载 Binance aggTrades CSV, 将时间戳转换为以 t_origin_date 00:00 UTC 为零点的秒数。

    使用 numpy 批量加载代替逐行 csv.reader, 大幅提升 I/O 性能。

    返回:
        trade_times : 交易到达时间戳 (秒, 相对于 t_origin)
        prices      : 交易价格
        quantities  : 交易数量
    """
    # 计算 origin 的 UNIX 微秒时间戳
    origin_dt = datetime.datetime.strptime(t_origin_date, '%Y-%m-%d')
    origin_dt = origin_dt.replace(tzinfo=datetime.timezone.utc)
    origin_us = int(origin_dt.timestamp() * 1_000_000)

    chunks_times = []
    chunks_prices = []
    chunks_qtys = []

    for csv_path in csv_paths:
        print('  加载: {} ...'.format(csv_path.name), end=' ', flush=True)
        import time as _t
        _t0 = _t.perf_counter()
        # aggTrades CSV 格式 (无表头):
        # 0: agg_trade_id, 1: price, 2: quantity,
        # 3: first_trade_id, 4: last_trade_id,
        # 5: timestamp (微秒), 6: is_buyer_maker, 7: is_best_match
        # 只读取需要的三列: price(1), quantity(2), timestamp(5)
        # np.loadtxt 在 numpy>=1.23 中使用 C 快速路径, 比 genfromtxt 快 5-10x
        data = np.loadtxt(
            str(csv_path), delimiter=',',
            usecols=(1, 2, 5),
            dtype=np.float64,
        )
        chunks_prices.append(data[:, 0])
        chunks_qtys.append(data[:, 1])
        # 时间戳: 微秒 → 相对秒
        chunks_times.append((data[:, 2] - origin_us) / 1_000_000.0)
        print('{:,d} trades ({:.1f}s)'.format(len(data), _t.perf_counter() - _t0),
              flush=True)

    trade_times = np.concatenate(chunks_times)
    prices = np.concatenate(chunks_prices)
    quantities = np.concatenate(chunks_qtys)

    # 确保时间单调递增
    sort_idx = np.argsort(trade_times)
    trade_times = trade_times[sort_idx]
    prices = prices[sort_idx]
    quantities = quantities[sort_idx]

    return trade_times, prices, quantities


def bin_trades(trade_times, dt, duration):
    """将交易时间戳装箱为计数序列。"""
    n_bins = int(round(duration / dt))
    counts = np.zeros(n_bins, dtype=int)
    t_centers = (np.arange(n_bins) + 0.5) * dt

    if len(trade_times) > 0:
        bin_idx = np.floor(trade_times / dt).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        # 使用 np.bincount 高效装箱
        bc = np.bincount(bin_idx, minlength=n_bins)
        counts[:] = bc[:n_bins]

    return counts, t_centers


# ====================================================================
# Task 1: 三体制预测对数似然
# ====================================================================

def run_task1_loglik(counts, t_centers, dt, duration):
    """
    对交易到达率执行三体制预测对数似然评估。

    返回:
        results : dict, 各方法各体制的归一化对数似然
    """
    print('\n[Task 1] 三体制预测对数似然')
    print('-' * 60)

    K = len(counts)

    # ---- 方法 1: Proposed Observer (1D 随机游走, 可预测强度 λ̂_{k|k-1}) ----
    obs = AdaptiveObserver1D(dt, FINANCE_R0, FINANCE_PHI, FINANCE_Q_SCALE,
                             FINANCE_ALPHA_G, EPS, P_INIT, enable_gating=True)
    obs.reset()
    lam_pred_obs = np.zeros(K)
    for k in range(K):
        # 1D 模型: F=1, 预测就是当前后验 (零阶保持, 无趋势外推)
        if obs.dormant:
            lam_pred_obs[k] = 0.0
        else:
            lam_pred_obs[k] = max(obs.x, 0.0)
        obs.step(k + 1, counts[k])

    # ---- 方法 2: EWMA (5s) ----
    ewma_5 = EWMAEstimator(tau=TAU_5, dt=dt)
    lam_ewma_5 = ewma_5.process_sequence(counts)
    lam_pred_ewma_5 = np.zeros(K)
    lam_pred_ewma_5[1:] = lam_ewma_5[:-1]

    # ---- 方法 3: EWMA (10s) ----
    ewma_fast = EWMAEstimator(tau=TAU_FAST, dt=dt)
    lam_ewma_fast = ewma_fast.process_sequence(counts)
    lam_pred_ewma_fast = np.zeros(K)
    lam_pred_ewma_fast[1:] = lam_ewma_fast[:-1]

    # ---- 方法 4: EWMA (15s) ----
    ewma_15 = EWMAEstimator(tau=TAU_15, dt=dt)
    lam_ewma_15 = ewma_15.process_sequence(counts)
    lam_pred_ewma_15 = np.zeros(K)
    lam_pred_ewma_15[1:] = lam_ewma_15[:-1]

    # ---- 方法 5: EWMA (慢, 300s) ----
    ewma_slow = EWMAEstimator(tau=TAU_SLOW, dt=dt)
    lam_ewma_slow = ewma_slow.process_sequence(counts)
    lam_pred_ewma_slow = np.zeros(K)
    lam_pred_ewma_slow[1:] = lam_ewma_slow[:-1]

    # ---- 方法 6: 标准 Kalman (常数噪声) — 使用显式 F@x 预测 ----
    avg_rate = np.mean(counts) / dt
    std_kf = make_standard_kf(dt, avg_rate, FINANCE_R0, FINANCE_PHI,
                              FINANCE_Q_SCALE, P_INIT)
    lam_pred_std = np.zeros(K)
    z = counts / dt
    x_kf = np.zeros(2)
    P_kf = std_kf.P_init.copy()
    for k in range(K):
        x_pred_kf = std_kf.F @ x_kf
        lam_pred_std[k] = max(x_pred_kf[0], 0.0)
        P_pred_kf = std_kf.F @ P_kf @ std_kf.F.T + std_kf.Q
        S_kf = (std_kf.H @ P_pred_kf @ std_kf.H.T).item() + std_kf.R
        K_kf = (P_pred_kf @ std_kf.H.T) / S_kf
        innov_kf = z[k] - (std_kf.H @ x_pred_kf).item()
        x_kf = x_pred_kf + K_kf.flatten() * innov_kf
        I_KH = np.eye(2) - K_kf @ std_kf.H
        P_kf = I_KH @ P_pred_kf @ I_KH.T + std_kf.R * (K_kf @ K_kf.T)
        x_kf[0] = max(x_kf[0], 0.0)

    # ---- 逐体制计算对数似然 ----
    methods = {
        'Proposed Observer': lam_pred_obs,
        'EWMA (5s)': lam_pred_ewma_5,
        'EWMA (10s)': lam_pred_ewma_fast,
        'EWMA (15s)': lam_pred_ewma_15,
        'EWMA (300s)': lam_pred_ewma_slow,
        'Standard KF': lam_pred_std,
    }

    regime_results = {}
    for regime_name, (t_lo, t_hi) in REGIME_BOUNDARIES.items():
        k_lo = max(int(t_lo / dt), 0)
        k_hi = min(int(t_hi / dt), K)
        n_regime = k_hi - k_lo

        regime_results[regime_name] = {}
        for method_name, lam_seq in methods.items():
            ll, _ = cumulative_predictive_loglik(
                lam_seq[k_lo:k_hi], counts[k_lo:k_hi], dt)
            regime_results[regime_name][method_name] = ll / max(n_regime, 1)

    # ---- 打印表格 (tab:finance_loglik) ----
    print('\n{:<25s} {:>12s} {:>12s} {:>12s}'.format(
        'Method', 'Pre-event', 'Event', 'Post-event'))
    print('-' * 65)
    for method_name in methods:
        vals = [regime_results[r][method_name] for r in
                ['pre_event', 'event', 'post_event']]
        print('{:<25s} {:12.4f} {:12.4f} {:12.4f}'.format(
            method_name, *vals))

    # ---- 绘制累积对数似然曲线 (fig:finance_loglik) ----
    _plot_cumulative_loglik(counts, methods, dt, duration)

    return regime_results


def _plot_cumulative_loglik(counts, methods, dt, duration):
    """绘制全程累积预测对数似然曲线 + 体制分割。"""
    _setup_mpl()
    K = len(counts)
    t_hours = (np.arange(K) + 0.5) * dt / 3600.0  # 转换为小时

    color_map = {
        'Proposed Observer': COLORS['proposed'],
        'EWMA (5s)': '#90CAF9',           # 浅蓝
        'EWMA (10s)': COLORS['ewma_fast'],
        'EWMA (15s)': '#CE93D8',          # 浅紫
        'EWMA (300s)': COLORS['ewma_slow'],
        'Standard KF': COLORS['standard_kf'],
    }

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    for method_name, lam_seq in methods.items():
        _, curve = cumulative_predictive_loglik(lam_seq, counts, dt)
        ax.plot(t_hours, curve, lw=1.2, label=method_name,
                color=color_map.get(method_name, 'gray'))

    # 体制分割线
    for regime_name, (t_lo, t_hi) in REGIME_BOUNDARIES.items():
        ax.axvline(t_lo / 3600, color='gray', ls='--', lw=0.6, alpha=0.5)

    # 标注体制
    ax.text(7.5, ax.get_ylim()[1] * 0.95, 'Pre-event',
            ha='center', fontsize=9, color='gray')
    ax.text(19.5, ax.get_ylim()[1] * 0.95, 'Event',
            ha='center', fontsize=9, color='#E53935', fontweight='bold')
    ax.text(36.0, ax.get_ylim()[1] * 0.95, 'Post-event',
            ha='center', fontsize=9, color='gray')

    ax.set_xlabel('Time [hours since Oct 10 00:00 UTC]')
    ax.set_ylabel('Cumulative predictive log-likelihood')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, ls=':', alpha=0.3)
    ax.set_title('Predictive Log-Likelihood: BTCUSDT 1011 Crash')

    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / 'finance_loglik.pdf'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    print('\n[Task 1] 对数似然图已保存: {}'.format(out_path))
    plt.close(fig)


# ====================================================================
# Task 2: 强度估计 + 自适应带宽可视化
# ====================================================================

def run_task2_intensity(counts, t_centers, dt, duration,
                        prices, trade_times):
    """
    可视化交易到达强度估计与自适应带宽:
      - 上面板: 价格 + 成交率
      - 下面板: Observer 有效带宽 τ_eff 的自适应轨迹
    """
    print('\n[Task 2] 强度估计与带宽自适应可视化')
    print('-' * 60)

    K = len(counts)

    # ---- Observer (1D 随机游走, 后验) 用于可视化 ----
    obs = AdaptiveObserver1D(dt, FINANCE_R0, FINANCE_PHI, FINANCE_Q_SCALE,
                             FINANCE_ALPHA_G, EPS, P_INIT, enable_gating=True)
    lam_obs = obs.process_sequence(counts)
    tau_eff = np.array(obs.history['tau_eff'])
    dormant = np.array(obs.history['dormant_flag'])

    # ---- EWMA 基线 ----
    ewma_fast = EWMAEstimator(tau=TAU_FAST, dt=dt)
    lam_fast = ewma_fast.process_sequence(counts)

    ewma_slow = EWMAEstimator(tau=TAU_SLOW, dt=dt)
    lam_slow = ewma_slow.process_sequence(counts)

    # ---- 价格以分钟级重采样 (减少绘图密度) ----
    t_hours = t_centers / 3600.0
    resample_factor = max(int(60 / dt), 1)  # 约 1 分钟分辨率
    idx_rs = np.arange(0, K, resample_factor)

    # 价格按分钟取中位数
    price_minute = np.zeros(len(idx_rs))
    for i, k_start in enumerate(idx_rs):
        k_end = min(k_start + resample_factor, K)
        t_lo = k_start * dt
        t_hi = k_end * dt
        mask = (trade_times >= t_lo) & (trade_times < t_hi)
        if np.any(mask):
            price_minute[i] = np.median(prices[mask])
        elif i > 0:
            price_minute[i] = price_minute[i - 1]

    # ---- 打印关键统计 ----
    crash_peak_k = np.argmax(lam_obs)
    crash_peak_t = crash_peak_k * dt / 3600
    min_tau = np.min(tau_eff[~dormant]) if np.any(~dormant) else dt
    max_tau = np.min([np.max(tau_eff[~dormant]), 1e6]) if np.any(~dormant) else dt

    print('  Observer 峰值强度: {:.1f} trades/s @ t={:.2f}h'.format(
        np.max(lam_obs), crash_peak_t))
    print('  有效带宽范围: {:.2f}s ~ {:.1f}s'.format(min_tau, max_tau))
    print('  休眠态 bins: {:,d} / {:,d} ({:.1f}%)'.format(
        int(np.sum(dormant)), K, np.sum(dormant) / K * 100))

    # ---- 绘制三面板图 ----
    _plot_intensity_and_bandwidth(
        t_hours, idx_rs, price_minute,
        lam_obs, lam_fast, lam_slow, tau_eff, dormant,
        resample_factor)

    return dict(
        peak_rate=float(np.max(lam_obs)),
        peak_time_h=crash_peak_t,
        min_tau=float(min_tau),
        max_tau=float(max_tau),
        dormant_frac=float(np.sum(dormant) / K),
    )


def _plot_intensity_and_bandwidth(t_hours, idx_rs, price_minute,
                                   lam_obs, lam_fast, lam_slow,
                                   tau_eff, dormant, resample_factor):
    """三面板图: 价格 / 强度估计 / 有效带宽。"""
    _setup_mpl()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1.5, 1]})

    t_hours_rs = t_hours[idx_rs]

    # ---- 上面板: 价格 ----
    ax0 = axes[0]
    ax0.plot(t_hours_rs, price_minute / 1000,
             color='black', lw=0.6, alpha=0.8)
    ax0.set_ylabel('BTC Price [K USD]')
    ax0.set_title('BTCUSDT 1011 Crash — Intensity Estimation & Bandwidth Adaptation')

    # 体制分割
    for _, (t_lo, t_hi) in REGIME_BOUNDARIES.items():
        ax0.axvline(t_lo / 3600, color='gray', ls='--', lw=0.5, alpha=0.4)

    # ---- 中面板: 强度估计 (对数标度) ----
    ax1 = axes[1]
    # 下采样用于绘图
    ax1.plot(t_hours[idx_rs], lam_obs[idx_rs],
             color=COLORS['proposed'], lw=0.8, label='Proposed Observer')
    ax1.plot(t_hours[idx_rs], lam_fast[idx_rs],
             color=COLORS['ewma_fast'], lw=0.6, alpha=0.6,
             label='EWMA ({}s)'.format(int(TAU_FAST)))
    ax1.plot(t_hours[idx_rs], lam_slow[idx_rs],
             color=COLORS['ewma_slow'], lw=0.6, alpha=0.6,
             label='EWMA ({}s)'.format(int(TAU_SLOW)))
    ax1.set_ylabel('Trade arrival rate [trades/s]')
    ax1.set_yscale('log')
    ax1.legend(fontsize=7, loc='upper right', ncol=3)
    ax1.grid(True, which='both', ls=':', alpha=0.2)

    for _, (t_lo, t_hi) in REGIME_BOUNDARIES.items():
        ax1.axvline(t_lo / 3600, color='gray', ls='--', lw=0.5, alpha=0.4)

    # ---- 下面板: 有效带宽 τ_eff ----
    ax2 = axes[2]
    tau_plot = tau_eff.copy()
    tau_plot[dormant] = np.nan  # 休眠态不显示
    ax2.plot(t_hours[idx_rs], tau_plot[idx_rs],
             color=COLORS['proposed'], lw=0.7)
    ax2.set_ylabel(r'Effective bandwidth $\tau_{\mathrm{eff}}$ [s]')
    ax2.set_xlabel('Time [hours since Oct 10 00:00 UTC]')
    ax2.set_yscale('log')
    ax2.grid(True, which='both', ls=':', alpha=0.2)

    # 水平参考线: EWMA 的固定带宽
    ax2.axhline(TAU_FAST, color=COLORS['ewma_fast'], ls='--', lw=0.6,
                alpha=0.5, label=r'$\tau_{{EWMA,fast}}$={}'.format(int(TAU_FAST)))
    ax2.axhline(TAU_SLOW, color=COLORS['ewma_slow'], ls='--', lw=0.6,
                alpha=0.5, label=r'$\tau_{{EWMA,slow}}$={}'.format(int(TAU_SLOW)))
    ax2.legend(fontsize=7, loc='upper right')

    for _, (t_lo, t_hi) in REGIME_BOUNDARIES.items():
        ax2.axvline(t_lo / 3600, color='gray', ls='--', lw=0.5, alpha=0.4)

    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / 'finance_intensity.pdf'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    print('\n[Task 2] 强度图已保存: {}'.format(out_path))
    plt.close(fig)


# ====================================================================
# 主入口
# ====================================================================

def run(save=True):
    """执行 Application II 全部任务。"""
    print('=' * 60)
    print('  Section 7.3: Application II — High-Frequency Finance')
    print('  BTCUSDT 1011 Crash (Oct 10-11, 2025)')
    print('=' * 60)

    # ---- 加载数据 ----
    print('\n[加载] aggTrades:')
    trade_times, prices, quantities = load_aggtrades(CSV_FILES)

    # duration 使用原点到末尾的绝对时长, 而非 first~last 的相对时长
    # 因为 trade_times 已相对于 Oct 10 00:00 UTC, 两天总长 = 48h = 172800s
    TOTAL_DURATION = 48 * 3600  # 172800 秒
    duration = TOTAL_DURATION
    print('  总 trades: {:,d}'.format(len(trade_times)))
    print('  时间跨度: {:.1f} 小时 ({:.0f} 秒)'.format(
        duration / 3600, duration))
    print('  价格范围: ${:,.0f} ~ ${:,.0f}'.format(
        np.min(prices), np.max(prices)))

    # ---- 装箱 ----
    counts, t_centers = bin_trades(trade_times, FINANCE_DT, duration)
    print('  Bin 宽度: {} s, 总 bins: {:,d}'.format(FINANCE_DT, len(counts)))
    print('  平均事件率: {:.1f} trades/s'.format(np.mean(counts) / FINANCE_DT))

    # ---- Task 1: 预测对数似然 ----
    task1_results = run_task1_loglik(counts, t_centers, FINANCE_DT, duration)

    # ---- Task 2: 强度可视化 ----
    task2_results = None
    if save:
        task2_results = run_task2_intensity(
            counts, t_centers, FINANCE_DT, duration, prices, trade_times)

    return dict(task1=task1_results, task2=task2_results)


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
    print('\n[App Finance] 全部任务完成。')
