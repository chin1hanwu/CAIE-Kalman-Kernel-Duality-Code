# SPDX-License-Identifier: Apache-2.0
"""
合成数据生成:
  Profile A — NHPP: 正弦包络 + 阶跃  (main.tex Eq. profile_a)
  Profile B — Hawkes 自激过程          (main.tex Eq. profile_b)
  Profile C — 间歇型 NHPP (结构零区)   (main.tex Eq. profile_c)
  Gamma 更新过程 (Experiment I 专用)

生成算法:
  NHPP  → Lewis–Shedler 稀疏化 (thinning)
  Hawkes → Ogata 修正稀疏化
"""

import numpy as np


# ====================================================================
# 强度函数 (ground truth)
# ====================================================================

def profile_a_intensity(t, params):
    """
    Profile A 的真实强度 λ_A(t)。
    λ_A(t) = λ_base + A·sin(2πft) + Σ_j Δλ_j · 1(t ≥ t_j)
    """
    lam = (params['lambda_base']
           + params['amplitude'] * np.sin(2.0 * np.pi * params['frequency'] * t))
    for tj, dj in zip(params['step_times'], params['step_magnitudes']):
        lam = lam + dj * (t >= tj)
    return np.maximum(lam, 0.0)


def profile_c_intensity(t, params):
    """
    Profile C 的真实强度 λ_C(t)。
    仅在活跃窗口 T_on 内为正, 其余为零。
    """
    t = np.asarray(t, dtype=float)
    lam_active = (params['lambda_base']
                  + params['amplitude'] * np.sin(2.0 * np.pi * params['frequency'] * t))
    lam_active = np.maximum(lam_active, 0.0)

    mask = np.zeros_like(t, dtype=bool)
    for t_start, t_end in params['active_windows']:
        mask |= (t >= t_start) & (t < t_end)

    return lam_active * mask


def hawkes_intensity(t_eval, events, mu, alpha, beta):
    """
    给定已有事件序列, 计算 Hawkes 过程在 t_eval 处的条件强度。
    λ(t) = μ + α·Σ_{t_i < t} β·exp(−β(t − t_i))
    """
    t_eval = np.asarray(t_eval, dtype=float)
    lam = np.full_like(t_eval, mu, dtype=float)
    for ti in events:
        contrib = alpha * beta * np.exp(-beta * np.maximum(t_eval - ti, 0.0))
        contrib *= (t_eval > ti)   # 因果: 仅 t > t_i
        lam += contrib
    return lam


# ====================================================================
# 事件生成
# ====================================================================

def generate_nhpp(lambda_func, lambda_max, duration, rng, params):
    """
    Lewis–Shedler 稀疏化算法生成 NHPP 事件。

    参数:
        lambda_func : 强度函数 λ(t, params)
        lambda_max  : 强度上界 (用于稀疏化)
        duration    : 仿真持续时间
        rng         : numpy RandomState
        params      : 传给 lambda_func 的参数字典

    返回:
        事件时间戳数组
    """
    events = []
    t = 0.0

    # 输入防御: lambda_max 必须为正
    if lambda_max <= 0:
        raise ValueError(f"lambda_max 必须为正数, 得到 {lambda_max}")

    while True:
        # 生成齐次 Poisson 候选间隔
        u = rng.exponential(1.0 / lambda_max)
        t += u
        if t >= duration:
            break
        # 稀疏化: 以概率 λ(t)/λ_max 接受
        lam_t = lambda_func(np.array([t]), params)[0]
        if rng.random() < lam_t / lambda_max:
            events.append(t)
    return np.array(events)


def generate_hawkes(mu, alpha, beta, duration, rng):
    """
    Ogata 修正稀疏化算法生成 Hawkes 过程事件。

    分支比 = α (核函数 h(s) = α·β·exp(−βs) 的积分)。
    要求 α < 1 (亚临界)。
    """
    # 输入防御: 亚临界检查
    if alpha >= 1.0:
        raise ValueError(f"分支比 alpha 必须 < 1 (亚临界), 得到 {alpha}")
    if mu <= 0:
        raise ValueError(f"背景强度 mu 必须为正数, 得到 {mu}")

    events = []
    t = 0.0
    lam_current = mu   # 当前强度

    while True:
        # 上界: 当前强度 (两事件间 λ(s) 单调递减)
        lam_bar = max(lam_current, mu)
        u = rng.exponential(1.0 / lam_bar)
        t += u
        if t >= duration:
            break

        # 衰减: λ(t+u) = μ + (λ(t) − μ)·exp(−β·u)
        lam_current = mu + (lam_current - mu) * np.exp(-beta * u)

        # 接受/拒绝
        if rng.random() < lam_current / lam_bar:
            events.append(t)
            # 新事件贡献: +α·β (即 h(0) = α·β)
            lam_current += alpha * beta

    return np.array(events)


def generate_gamma_renewal(kappa, lam, n_events, rng):
    """
    生成 Gamma(κ, κλ) 更新过程。
    均值间隔 = 1/λ, 形状参数 κ 控制变异性。

    特殊情况:
        κ = 1  → Poisson
        κ < 1  → 突发态 (高变异)
        κ → ∞  → 确定性 (等间隔)
    """
    # Gamma(κ, κλ) 的均值 = 1/λ, 方差 = 1/(κλ²)
    # numpy 的 gamma(shape, scale), 其中 shape=κ, scale=1/(κλ)
    scale = 1.0 / (kappa * lam)
    iat = rng.gamma(shape=kappa, scale=scale, size=n_events)
    events = np.cumsum(iat)
    return events


# ====================================================================
# 辅助工具
# ====================================================================

def bin_events(events, dt, duration):
    """
    将事件时间戳装箱为 bin 计数。

    返回:
        counts : 长度 K 的整数数组, K = round(duration / dt)
        t_bins : bin 中心时间
    """
    n_bins = int(round(duration / dt))
    counts = np.zeros(n_bins, dtype=int)
    t_bins = (np.arange(n_bins) + 0.5) * dt

    if len(events) == 0:
        return counts, t_bins

    # 将事件映射到 bin 索引
    bin_idx = np.floor(events / dt).astype(int)
    # 越界检测: 警告而非静默裁剪
    n_oob = int(np.sum((bin_idx < 0) | (bin_idx >= n_bins)))
    if n_oob > 0:
        import warnings
        warnings.warn(
            f"bin_events: {n_oob} 个事件超出 [0, {duration}) 范围, 已裁剪到边界 bin",
            RuntimeWarning, stacklevel=2)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    for idx in bin_idx:
        counts[idx] += 1

    return counts, t_bins


def compute_lambda_max(lambda_func, params, duration, n_sample=10000):
    """
    通过密集采样估计强度函数的上界 λ_max。
    用于 Lewis–Shedler 稀疏化算法。
    """
    t_sample = np.linspace(0, duration, n_sample)
    lam_vals = lambda_func(t_sample, params)
    return float(np.max(lam_vals)) * 1.1   # 加 10% 安全裕量
