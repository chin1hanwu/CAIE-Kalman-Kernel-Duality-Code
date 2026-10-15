# SPDX-License-Identifier: Apache-2.0
"""
基线估计器:
  1. 倒数间隔估计器 (1/δt)
  2. 固定带宽 EWMA (快 / 慢)
  3. 标准 Kalman 滤波器 (常数 Q, R)
  4. 离线自适应 KDE — Abramson 平方根律 (Oracle 上界, 非因果)
"""

import numpy as np


# ====================================================================
# 1. 倒数间隔估计器
# ====================================================================

def reciprocal_on_bins(events, dt, duration):
    """
    基于 1/δt_i 的倒数估计, 映射到 bin 网格 (严格因果: 零阶保持)。

    在每个 bin 中心 tc, 使用 tc 之前最近完成的间隔的倒数。
    不偷看任何尚未发生的事件。

    参数:
        events   : 事件时间戳数组
        dt       : bin 宽度
        duration : 仿真总时长

    返回:
        长度 K 的 bin 网格强度估计
    """
    n_bins = int(round(duration / dt))
    t_centers = (np.arange(n_bins) + 0.5) * dt
    lam_hat = np.zeros(n_bins)

    if len(events) < 2:
        return lam_hat

    iat = np.diff(events)                          # 间隔 δt_i
    reciprocals = 1.0 / np.maximum(iat, 1e-12)

    # 严格因果: 在 tc 时刻, 最近完成的间隔索引 = searchsorted(events, tc) - 2
    # (扣 1 得到已发生的最后一个事件索引, 再扣 1 得到该事件完成的间隔索引)
    event_idx = np.searchsorted(events, t_centers, side='right') - 1

    for i, idx in enumerate(event_idx):
        # idx 是 tc 之前最后一个已发生事件的索引
        # 该事件完成的间隔索引为 idx - 1 (需要至少两个事件才有一个间隔)
        iat_idx = idx - 1
        if 0 <= iat_idx < len(iat):
            lam_hat[i] = reciprocals[iat_idx]
        # 否则 (不足两个事件): 保持 0

    return lam_hat


def reciprocal_mean_bias(events):
    """
    计算一组事件的倒数估计均值 E[1/δt]。
    用于 Experiment I 的偏差比分析。
    """
    if len(events) < 2:
        return np.nan
    iat = np.diff(events)
    return np.mean(1.0 / np.maximum(iat, 1e-12))


# ====================================================================
# 2. EWMA 估计器
# ====================================================================

class EWMAEstimator:
    """
    指数加权移动平均强度估计器。
    等价于固定带宽因果指数核: K_τ(u) = (1/τ)·exp(−u/τ), u ≥ 0。
    平滑参数 α = 1 − exp(−Δt/τ)。
    """

    def __init__(self, tau, dt):
        self.tau = tau
        self.dt = dt
        self.alpha = 1.0 - np.exp(-dt / tau)

    def process_sequence(self, counts):
        """处理 bin 计数序列, 返回强度估计 (events/s)。"""
        z = counts / self.dt
        n = len(z)
        lam_hat = np.zeros(n)
        for k in range(n):
            if k == 0:
                lam_hat[k] = self.alpha * z[k]
            else:
                lam_hat[k] = (1.0 - self.alpha) * lam_hat[k - 1] + self.alpha * z[k]
        return lam_hat


# ====================================================================
# 3. 标准 Kalman 滤波器 (常数噪声参数)
# ====================================================================

class StandardKalmanFilter:
    """
    结构与 AdaptiveObserver 相同的 CV 模型,
    但 Q 和 R 为常数 (非内生), 不含休眠态门控。
    用于消融实验: 隔离内生噪声参数化的贡献。
    """

    def __init__(self, dt, Q_const, R_const, P_init):
        """
        参数:
            Q_const : 常数过程噪声协方差 (2×2)
            R_const : 常数测量噪声方差 (标量)
        """
        self.dt = dt
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = Q_const.copy()
        self.R = float(R_const)
        self.P_init = P_init.copy()

    def process_sequence(self, counts):
        """处理 bin 计数序列, 返回强度估计。"""
        z = counts / self.dt
        n = len(z)
        lam_hat = np.zeros(n)

        x = np.zeros(2)
        P = self.P_init.copy()

        for k in range(n):
            # 预测
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q

            # 更新
            S = (self.H @ P_pred @ self.H.T).item() + self.R
            K = (P_pred @ self.H.T) / S
            innov = z[k] - (self.H @ x_pred).item()
            x = x_pred + K.flatten() * innov
            # Joseph 形式协方差更新 (数值更稳定, 保持 PSD)
            I_KH = np.eye(2) - K @ self.H
            P = I_KH @ P_pred @ I_KH.T + self.R * (K @ K.T)

            # 非负投影
            x[0] = max(x[0], 0.0)
            lam_hat[k] = x[0]

        return lam_hat


def make_standard_kf(dt, lambda_avg, R0, phi, q, P_init):
    """
    使用平均强度 λ_avg 构造"最优调参"的标准 KF。
    这是常数噪声 KF 能做到的最佳近似。
    """
    G = np.array([
        [dt**3 / 3.0, dt**2 / 2.0],
        [dt**2 / 2.0, dt],
    ])
    Q_const = q * lambda_avg**2 * G
    R_const = R0 + phi * lambda_avg / dt
    return StandardKalmanFilter(dt, Q_const, R_const, P_init)


# ====================================================================
# 4. 离线自适应 KDE (Oracle, 非因果, O(N²))
# ====================================================================

class OracleAdaptiveKDE:
    """
    Abramson (1982) 两遍自适应带宽 KDE。
    第一遍: 固定带宽高斯核 → pilot 强度估计
    第二遍: h_i = h₀ · (λ_pilot(t_i) / g)^{−1/2}
    对称高斯核, 使用全局数据 (非因果)。
    作为任何因果滤波器的理论性能上界。
    """

    def __init__(self, h_pilot=1.0):
        self.h_pilot = h_pilot

    def estimate(self, events, t_eval, chunk_size=500):
        """
        在 t_eval 网格点处估计强度。
        使用分块计算防止 N×N 矩阵导致内存溢出。

        参数:
            events     : 事件时间戳数组
            t_eval     : 评估时间点数组
            chunk_size : 分块大小, 控制峰值内存

        返回:
            强度估计值数组 (与 t_eval 等长)
        """
        N = len(events)
        if N == 0:
            return np.zeros(len(t_eval))

        if N == 1:
            diff = t_eval - events[0]
            lam_hat = np.exp(-0.5 * (diff / self.h_pilot)**2) / (
                self.h_pilot * np.sqrt(2.0 * np.pi))
            return lam_hat

        # --- 第一遍: 固定带宽高斯核 pilot (分块, 防 OOM) ---
        inv_h = 1.0 / self.h_pilot
        norm_const = inv_h / np.sqrt(2.0 * np.pi)
        pilot = np.zeros(N)
        # np.errstate: 抑制 exp underflow 警告 (远距离事件对的高斯核精确为 0)
        with np.errstate(under='ignore'):
            for i0 in range(0, N, chunk_size):
                i1 = min(i0 + chunk_size, N)
                diff_chunk = events[i0:i1, np.newaxis] - events[np.newaxis, :]
                pilot[i0:i1] = np.sum(
                    np.exp(-0.5 * (diff_chunk * inv_h)**2), axis=1
                ) * norm_const
        pilot = np.maximum(pilot, 1e-10)

        g = np.exp(np.mean(np.log(pilot)))

        # --- 第二遍: Abramson 自适应带宽 ---
        h_adapt = self.h_pilot * (pilot / g) ** (-0.5)

        K = len(t_eval)
        lam_hat = np.zeros(K)
        with np.errstate(under='ignore'):
            for i0 in range(0, K, chunk_size):
                i1 = min(i0 + chunk_size, K)
                diff_eval = t_eval[i0:i1, np.newaxis] - events[np.newaxis, :]
                u = diff_eval / h_adapt[np.newaxis, :]
                kernel_vals = np.exp(-0.5 * u**2) / (
                    h_adapt[np.newaxis, :] * np.sqrt(2.0 * np.pi))
                lam_hat[i0:i1] = np.sum(kernel_vals, axis=1)
        return lam_hat
