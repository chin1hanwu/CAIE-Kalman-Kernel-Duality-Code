# SPDX-License-Identifier: Apache-2.0
"""
评估指标 (对应 main.tex Sec 6.1 中的公式定义):
  - RMSE      (Eq. metrics_accuracy)
  - MISE      (Eq. metrics_accuracy)
  - 互相关峰值滞后 l*  (Eq. metric_lag)
  - 归一化泄漏质量 M̄_leak  (Eq. metric_leakage)
"""

import numpy as np


def rmse(lam_true, lam_hat):
    """
    均方根误差 (Eq. metrics_accuracy 左式)。
    RMSE = sqrt( (1/K) Σ (λ̂_k − λ_k)² )
    """
    return np.sqrt(np.mean((lam_hat - lam_true) ** 2))


def mise(lam_true, lam_hat, dt):
    """
    均值积分平方误差 (Eq. metrics_accuracy 右式)。
    MISE = ∫₀ᵀ (λ̂(t) − λ(t))² dt ≈ Σ (λ̂_k − λ_k)² · Δt
    """
    return np.sum((lam_hat - lam_true) ** 2) * dt


def cross_correlation_lag(lam_true, lam_hat, dt):
    """
    互相关峰值滞后 l* (Eq. metric_lag)。
    l* = argmax_l { Σ_k λ̂_{k+l} · λ_k / (||λ̂|| · ||λ||) }

    正值表示估计滞后于真值 (相位延迟)。
    返回滞后量 (秒)。
    """
    # 去均值
    a = lam_true - np.mean(lam_true)
    b = lam_hat - np.mean(lam_hat)

    # 归一化互相关
    norm = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if norm < 1e-12:
        return 0.0

    xcorr = np.correlate(b, a, mode='full')
    xcorr /= norm

    # 互相关的零滞后位于索引 len(a) - 1
    zero_lag_idx = len(a) - 1
    peak_idx = np.argmax(xcorr)
    lag_bins = peak_idx - zero_lag_idx
    return lag_bins * dt


def normalized_leakage_mass(lam_hat, zero_mask, dt):
    """
    归一化泄漏质量 (Eq. metric_leakage)。
    M̄_leak = (1/|T_zero|) ∫_{T_zero} λ̂(t) dt

    参数:
        lam_hat   : 估计强度序列
        zero_mask : 布尔数组, True 表示该 bin 处于结构零区
        dt        : bin 宽度

    返回:
        每单位休眠时间的平均泄漏率 (events/s)
    """
    T_zero = np.sum(zero_mask) * dt
    if T_zero < 1e-12:
        return 0.0
    leak_integral = np.sum(lam_hat[zero_mask]) * dt
    return leak_integral / T_zero


def wake_up_lag(lam_hat, lam_true, wake_idx, dt, threshold_frac=0.5):
    """
    唤醒滞后: 从休眠→活跃切换点到估计强度恢复至真值一定比例所需的时间步数。

    参数:
        wake_idx       : 唤醒 bin 的索引
        threshold_frac : 达标比例 (默认 50%)

    返回:
        滞后步数
    """
    if wake_idx >= len(lam_hat):
        return np.nan
    target = lam_true[wake_idx] * threshold_frac

    for offset in range(len(lam_hat) - wake_idx):
        if lam_hat[wake_idx + offset] >= target:
            return offset
    return len(lam_hat) - wake_idx   # 未达标
