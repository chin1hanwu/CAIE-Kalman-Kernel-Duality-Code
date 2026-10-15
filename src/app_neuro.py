# SPDX-License-Identifier: Apache-2.0
"""基于本地 MC_Maze NWB 数据的神经元发放率分析。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt

from config import (DT, R0, PHI, Q_SCALE, ALPHA_G, EPS, P_INIT,
                    COLORS, FIGSIZE_SINGLE, DPI, FIGURE_DIR)
from observer import AdaptiveObserver
from baselines import EWMAEstimator
from data_gen import bin_events
from metrics_realworld import (
    time_rescaling_from_bins, ks_test_exponential,
    qq_data_exponential, cumulative_predictive_loglik,
)


# ====================================================================
# 配置
# ====================================================================

NWB_PATH = Path(__file__).parent.parent / 'data' / 'neuro' / \
    'sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb'

# 选取用于分析的神经元数量 (取发放率最高的 top-N 个)
N_UNITS_QQ = 6         # Q-Q 图中展示的代表性神经元数量

# PSTH 基线: 固定 bin 宽度 (秒)
PSTH_BIN_WIDTH = 0.05  # 50 ms, 标准 PSTH 分辨率

# 分析时间窗口: 取一段连续试次 (避免过长的间歇)
ANALYSIS_DURATION = 600.0   # 分析前 600 秒 (10 分钟)



# ====================================================================
# 数据加载
# ====================================================================

def load_mc_maze(nwb_path, duration=None):
    """
    从 NWB 文件加载 MC_Maze 数据。

    返回:
        spike_trains : list of numpy arrays, 各神经元的 spike 时间戳
        unit_ids     : 神经元 ID 列表
        session_dur  : 会话总时长 (秒)
    """
    io = NWBHDF5IO(str(nwb_path), 'r')
    nwb = io.read()

    n_units = len(nwb.units)
    spike_trains = []
    for i in range(n_units):
        spikes = nwb.units['spike_times'][i]
        spike_trains.append(np.asarray(spikes))

    unit_ids = nwb.units.id[:].tolist()

    # 限制分析时长
    if duration is not None:
        spike_trains = [s[s < duration] for s in spike_trains]

    session_dur = duration if duration is not None else spike_trains[-1][-1]

    io.close()
    return spike_trains, unit_ids, session_dur


def select_top_units(spike_trains, n_select, duration):
    """选取发放率最高的 n_select 个神经元, 返回索引列表。"""
    rates = np.array([len(s) / duration for s in spike_trains])
    top_idx = np.argsort(rates)[::-1][:n_select]
    return top_idx.tolist()


# ====================================================================
# PSTH 基线 (固定 bin 宽度高斯平滑)
# ====================================================================

def psth_estimate(spike_times, dt, duration, sigma=None):
    """
    标准 PSTH: 将 spike 装箱后用高斯核平滑。
    sigma 默认为 bin_width * 2。
    """
    if sigma is None:
        sigma = dt * 2.0

    n_bins = int(round(duration / dt))
    counts = np.zeros(n_bins)
    t_centers = (np.arange(n_bins) + 0.5) * dt

    # 装箱
    if len(spike_times) > 0:
        bin_idx = np.floor(spike_times / dt).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        for idx in bin_idx:
            counts[idx] += 1

    # 因果高斯平滑: 仅使用过去数据 (半高斯, 向左截断)
    rate = counts / dt
    sigma_bins = sigma / dt
    # 因果实现: 手动构造半高斯核 (仅使用过去 + 当前)
    kernel_radius = int(4 * sigma_bins)
    kernel_t = np.arange(-kernel_radius, 1)  # 负方向 + 当前
    kernel = np.exp(-0.5 * (kernel_t / sigma_bins) ** 2)
    kernel /= kernel.sum()  # 归一化
    # np.convolve 要求核的 index 0 对应当前时刻的权重,
    # 所以翻转: kernel[0]=当前(最大), kernel[-1]=最远过去(最小)
    kernel = kernel[::-1]
    smoothed = np.convolve(rate, kernel, mode='full')[:len(rate)]
    return np.maximum(smoothed, 0.0), t_centers


# ====================================================================
# Task 1: 时间重缩放 Q-Q 图与 KS 检验
# ====================================================================

def run_task1_time_rescaling(spike_trains, unit_indices, duration, save=True):
    """
    对选定神经元执行时间重缩放检验。

    返回:
        results : dict, 各方法的 KS 统计量列表
    """
    n_bins = int(round(duration / DT))

    results_observer = []
    results_psth = []
    qq_data = {}

    print('\n[Task 1] 时间重缩放 Q-Q 图 & KS 检验')
    print('-' * 60)

    for rank, uid in enumerate(unit_indices):
        spikes = spike_trains[uid]
        if len(spikes) < 20:
            continue

        # 装箱
        counts, t_bins = bin_events(spikes, DT, duration)

        # --- Observer 估计: 使用可预测强度 λ̂_{k|k-1} (CRITICAL) ---
        # time-rescaling theorem 要求使用 predictable intensity,
        # 即 update 之前的 predict 值, 否则 KS 检验失去理论效力。
        obs = AdaptiveObserver(DT, R0, PHI, Q_SCALE, ALPHA_G, EPS,
                               P_INIT, enable_gating=True)
        obs.reset()
        K = len(counts)
        lam_obs = np.zeros(K)  # λ̂_{k|k-1} (可预测强度, 含休眠门控)
        for k in range(K):
            # 如果 Observer 已处于休眠态, 预测强度为 0
            if obs.dormant:
                lam_obs[k] = 0.0
            else:
                x_pred = obs.F @ obs.x
                lam_obs[k] = max(x_pred[0], 0.0)
            obs.step(k + 1, counts[k])

        # --- PSTH 基线 (因果高斯平滑) ---
        lam_psth, _ = psth_estimate(spikes, PSTH_BIN_WIDTH, duration)
        # 重采样到 DT 网格
        t_psth = (np.arange(len(lam_psth)) + 0.5) * PSTH_BIN_WIDTH
        lam_psth_resampled = np.interp(t_bins, t_psth, lam_psth)

        # --- 时间重缩放 ---
        tau_obs = time_rescaling_from_bins(spikes, lam_obs, DT, duration)
        tau_psth = time_rescaling_from_bins(spikes, lam_psth_resampled, DT, duration)

        # --- KS 检验 ---
        ks_obs, p_obs = ks_test_exponential(tau_obs)
        ks_psth, p_psth = ks_test_exponential(tau_psth)

        results_observer.append(dict(unit=uid, ks=ks_obs, p=p_obs,
                                     n_spikes=len(spikes)))
        results_psth.append(dict(unit=uid, ks=ks_psth, p=p_psth,
                                 n_spikes=len(spikes)))

        # 保存 Q-Q 数据 (仅前 N_UNITS_QQ 个)
        if rank < N_UNITS_QQ:
            theo_obs, emp_obs = qq_data_exponential(tau_obs)
            theo_psth, emp_psth = qq_data_exponential(tau_psth)
            qq_data[uid] = dict(
                obs_theo=theo_obs, obs_emp=emp_obs,
                psth_theo=theo_psth, psth_emp=emp_psth,
            )

        rate = len(spikes) / duration
        print('  Unit {:4d}: rate={:.1f} Hz, '
              'KS(Observer)={:.4f} (p={:.3f}), '
              'KS(PSTH)={:.4f} (p={:.3f})'.format(
                  uid, rate, ks_obs, p_obs, ks_psth, p_psth))

    # ---- 汇总表 (tab:neuro_ks) ----
    ks_obs_vals = [r['ks'] for r in results_observer if not np.isnan(r['ks'])]
    ks_psth_vals = [r['ks'] for r in results_psth if not np.isnan(r['ks'])]
    p_obs_pass = sum(1 for r in results_observer if r['p'] > 0.05)
    p_psth_pass = sum(1 for r in results_psth if r['p'] > 0.05)
    n_total = len(results_observer)

    if not ks_obs_vals:
        print('\n[Task 1] 警告: 所有神经元被过滤, 无 KS 检验结果')
        return dict(observer=results_observer, psth=results_psth)

    print('\n[Task 1] ---- KS 检验汇总 (tab:neuro_ks) ----')
    print('{:<25s} {:>12s} {:>12s} {:>18s}'.format(
        'Method', 'Median D_N', 'Max D_N', 'Frac p>0.05'))
    print('-' * 70)
    print('{:<25s} {:12.4f} {:12.4f} {:12d}/{}'.format(
        'Proposed Observer',
        np.median(ks_obs_vals), np.max(ks_obs_vals),
        p_obs_pass, n_total))
    print('{:<25s} {:12.4f} {:12.4f} {:12d}/{}'.format(
        'PSTH (50ms)',
        np.median(ks_psth_vals), np.max(ks_psth_vals),
        p_psth_pass, n_total))

    # ---- 绘制 Q-Q 图 (fig:neuro_qq) ----
    if save and qq_data:
        _plot_qq(qq_data)

    return dict(observer=results_observer, psth=results_psth)


def _plot_qq(qq_data):
    """绘制多神经元 Q-Q 图。"""
    _setup_mpl()
    n_panels = len(qq_data)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, (uid, data) in enumerate(qq_data.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        # 参考线
        max_val = max(data['obs_theo'][-1] if len(data['obs_theo']) else 1,
                      data['psth_theo'][-1] if len(data['psth_theo']) else 1)
        ax.plot([0, max_val * 1.1], [0, max_val * 1.1],
                'k--', lw=0.8, alpha=0.5, label='Ideal')

        # Observer
        ax.scatter(data['obs_theo'], data['obs_emp'],
                   s=4, alpha=0.4, color=COLORS['proposed'],
                   label='Proposed', zorder=3, rasterized=True)

        # PSTH
        ax.scatter(data['psth_theo'], data['psth_emp'],
                   s=4, alpha=0.3, color='#9E9E9E',
                   label='PSTH', zorder=2, rasterized=True)

        ax.set_xlabel('Exp(1) quantiles')
        ax.set_ylabel('Empirical quantiles')
        ax.set_title('Unit {}'.format(uid), fontsize=10)
        ax.set_aspect('equal')
        if idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    # 隐藏多余面板
    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle('Time-Rescaling Q-Q Plots (Application I)', fontsize=12)
    fig.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / 'neuro_qq.pdf'
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    print('\n[Task 1] Q-Q 图已保存: {}'.format(out_path))
    plt.close(fig)


# ====================================================================
# Task 3: 预测对数似然
# ====================================================================

def run_task3_loglik(spike_trains, unit_indices, duration):
    """
    计算代表性神经元的一步预测对数似然。
    """
    print('\n[Task 3] 预测对数似然')
    print('-' * 60)

    results = {}
    for uid in unit_indices[:10]:  # 取前 10 个
        spikes = spike_trains[uid]
        counts, t_bins = bin_events(spikes, DT, duration)

        # Observer: 使用预测值 (即 predict 步之后、update 步之前的值)
        obs = AdaptiveObserver(DT, R0, PHI, Q_SCALE, ALPHA_G, EPS,
                               P_INIT, enable_gating=True)
        obs.reset()
        K = len(counts)
        lam_pred = np.zeros(K)  # λ̂_{k|k-1} (含休眠门控)
        for k in range(K):
            if obs.dormant:
                lam_pred[k] = 0.0
            else:
                x_pred = obs.F @ obs.x
                lam_pred[k] = max(x_pred[0], 0.0)
            obs.step(k + 1, counts[k])

        # EWMA
        ewma = EWMAEstimator(tau=0.1, dt=DT)
        lam_ewma = ewma.process_sequence(counts)
        # EWMA 的 "预测" 就是上一步的后验
        lam_ewma_pred = np.zeros(K)
        lam_ewma_pred[1:] = lam_ewma[:-1]

        # 计算累积对数似然
        ll_obs, _ = cumulative_predictive_loglik(lam_pred, counts, DT)
        ll_ewma, _ = cumulative_predictive_loglik(lam_ewma_pred, counts, DT)

        results[uid] = dict(
            ll_observer=ll_obs / K,   # 归一化到每步
            ll_ewma=ll_ewma / K,
        )

    # 汇总
    obs_lls = [v['ll_observer'] for v in results.values()]
    ewma_lls = [v['ll_ewma'] for v in results.values()]
    print('{:<25s} {:>12s}'.format('Method', 'Mean ℓ/step'))
    print('-' * 40)
    print('{:<25s} {:12.4f}'.format('Proposed Observer', np.mean(obs_lls)))
    print('{:<25s} {:12.4f}'.format('EWMA (100ms)', np.mean(ewma_lls)))

    return results


# ====================================================================
# 主入口
# ====================================================================

def run(save=True):
    """执行 Application I 全部任务。"""
    print('=' * 60)
    print('  Section 7.2: Application I — Computational Neuroscience')
    print('=' * 60)

    # 加载数据
    print('\n[加载] MC_Maze NWB: {}'.format(NWB_PATH))
    spike_trains, unit_ids, session_dur = \
        load_mc_maze(NWB_PATH, duration=ANALYSIS_DURATION)

    n_units = len(spike_trains)
    total_spikes = sum(len(s) for s in spike_trains)
    print('  神经元: {} 个, 总 spikes: {}, 时长: {:.0f} s'.format(
        n_units, total_spikes, session_dur))

    # 选取高发放率神经元
    top_units = select_top_units(spike_trains, N_UNITS_QQ,
                                 session_dur)
    print('  选取 top-{} 高发放率神经元'.format(len(top_units)))

    # Task 1: Time-Rescaling
    task1_results = run_task1_time_rescaling(
        spike_trains, top_units[:N_UNITS_QQ], session_dur, save=save)

    # Task 3: Predictive Log-Likelihood
    task3_results = run_task3_loglik(
        spike_trains, top_units, session_dur)

    return dict(task1=task1_results, task3=task3_results)


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
    print('\n[App Neuro] 全部任务完成。')
