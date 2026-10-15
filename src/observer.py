# SPDX-License-Identifier: Apache-2.0
"""
自适应核强度观测器 (Adaptive Kernel-Intensity Observer)。

包含两个模型:
  1. AdaptiveObserver    — 2D CV 模型 (Algorithm 1, 工程实现)
  2. AdaptiveObserver1D  — 1D 标量随机游走 (理论验证, 严格匹配 Theorem 1)

2D CV 模型:
  状态向量:  x_k = [λ_k, λ̇_k]^T               (Eq. state_vector)
  转移矩阵:  F = [[1, Δt], [0, 1]]              (Eq. transition)
  观测矩阵:  H = [1, 0]                          (Eq. observation)
  测量噪声:  R_k = R_0 + φ·max(λ̂, ε) / Δt       (Eq. R_model)
  过程噪声:  Q_k = q·max(λ̂², ε)·G               (Eq. Q_model)

1D 标量模型:
  状态:      x_k = λ_k (标量)
  转移:      F = 1 (随机游走)
  测量噪声:  R_k = R_0 + φ·λ / Δt
  过程噪声:  Q_k = q·λ²·Δt
  稳态增益:  K_ss ∝ λ^{1/2}  ⟹  τ_eff ∝ λ^{-1/2} (Theorem 1)
"""

import numpy as np


# ====================================================================
# 1D 标量随机游走观测器 (理论验证用)
# ====================================================================

class AdaptiveObserver1D:
    """
    1D 标量随机游走自适应观测器。

    严格实现 Theorem 1 (thm:sqrt_gain) 和 Corollary 1 (cor:bandwidth)
    的标量特化版本。稳态 Kalman 增益 K_ss = sqrt(qΔt/φ) · sqrt(λ),
    有效带宽 τ_eff = sqrt(φΔt/(qλ)) ∝ λ^{-1/2}, 与 Abramson 平方根
    法则一致。

    仅用于 Experiment II 的理论缩放律验证。
    实际应用 (Section 7) 请使用 2D CV 版本 AdaptiveObserver。
    """

    def __init__(self, dt, R0, phi, q, alpha_g, eps, P_init,
                 enable_gating=True, innov_clip=-1.0, T_gap_min=1.0):
        self.dt = dt
        self.R0 = R0
        self.phi = phi
        self.q = q
        self.alpha_g = alpha_g
        self.eps = eps
        self.enable_gating = enable_gating
        self.innov_clip = innov_clip   # 新息饱和阈值 γ (标准差倍数, ≤0 禁用)
        self._T_gap_min = T_gap_min
        # 兼容 2D P_init 矩阵: 取 (0,0) 元素
        if isinstance(P_init, np.ndarray):
            self.P_init = float(P_init[0, 0])
        else:
            self.P_init = float(P_init)
        self.reset()

    def reset(self):
        """重置滤波器到初始状态。"""
        self.x = 0.0                  # 标量强度 λ̂
        self.P = self.P_init          # 标量协方差
        self.t_last = 0.0
        self.dormant = False
        self._just_woke = True        # 首步视同唤醒
        self._lam_anchor = self.eps
        self.history = dict(
            lambda_hat=[], K_gain=[], tau_eff=[], dormant_flag=[],
        )

    def step(self, k, dN_k):
        """执行单步 predict-update 循环 (1D 标量版)。"""
        z_k = dN_k / self.dt

        # 事件到达与唤醒
        if dN_k > 0:
            self.t_last = k * self.dt
            if self.dormant:
                self.dormant = False
                self._just_woke = True
                self.P = self.P_init

        # 休眠态
        if self.dormant:
            self.x = 0.0
            self._record(0.0, 0.0, np.inf, True)
            return 0.0

        # 间隙检测
        if self.enable_gating:
            T_silent = k * self.dt - self.t_last
            T_gap = -np.log(self.alpha_g) / self._lam_anchor
            T_gap = max(T_gap, self._T_gap_min)
            if T_silent > T_gap:
                self.dormant = True
                self.x = 0.0
                self._record(0.0, 0.0, np.inf, True)
                return 0.0

        # 内生噪声 (1D 标量版)
        lam_prev = max(self.x, self.eps)
        # 唤醒态 R 保护: 刚唤醒时 λ̂=0, 用 z_k 作为 R 的异方差参考
        if self._just_woke and z_k > 0:
            lam_for_R = z_k
            self._just_woke = False
        else:
            lam_for_R = lam_prev
            self._just_woke = False
        R_k = self.R0 + self.phi * lam_for_R / self.dt
        Q_k = self.q * (lam_prev ** 2) * self.dt    # 随机游走增量方差

        # 预测 (F=1, x_pred = x)
        P_pred = self.P + Q_k

        # 更新
        S_k = P_pred + R_k
        K_k = P_pred / S_k
        innov = z_k - self.x

        # 新息饱和: 抑制 Poisson 计数噪声的极端离群值
        if self.innov_clip > 0 and S_k > 0:
            innov_std = innov / np.sqrt(S_k)
            if abs(innov_std) > self.innov_clip:
                innov = self.innov_clip * np.sqrt(S_k) * np.sign(innov)

        self.x = self.x + K_k * innov
        # Joseph 形式 (标量): P = (1-K)²P_pred + K²R
        self.P = (1.0 - K_k) ** 2 * P_pred + K_k ** 2 * R_k
        self.x = max(self.x, 0.0)

        # 锚定强度
        if dN_k > 0:
            self._lam_anchor = max(self.x, self.eps)

        # 有效带宽
        if 0.0 < K_k < 1.0:
            tau = -self.dt / np.log(1.0 - K_k)
        else:
            tau = self.dt / max(K_k, self.eps)

        self._record(self.x, K_k, tau, False)
        return self.x

    def process_sequence(self, counts):
        """处理完整的 bin 计数序列。"""
        self.reset()
        K = len(counts)
        estimates = np.zeros(K)
        for k in range(K):
            estimates[k] = self.step(k + 1, counts[k])
        return estimates

    def _record(self, lam, K, tau, dormant):
        self.history['lambda_hat'].append(lam)
        self.history['K_gain'].append(K)
        self.history['tau_eff'].append(tau)
        self.history['dormant_flag'].append(dormant)


# ====================================================================
# 2D CV 观测器 (工程实现, Algorithm 1)
# ====================================================================



class AdaptiveObserver:
    """
    完全因果、O(1) 复杂度的自适应核强度观测器。
    """

    def __init__(self, dt, R0, phi, q, alpha_g, eps, P_init,
                 enable_gating=True, innov_clip=-1.0, T_gap_min=1.0):
        """
        参数:
            dt         : 离散时间步长 Δt
            R0         : 测量噪声基线方差 R_0
            phi        : Fano 因子 φ (≥ 1)
            q          : 过程噪声缩放系数 q
            alpha_g    : 休眠态检测显著性水平 α_g
            eps        : 数值正则化常数 ε
            P_init     : 初始协方差矩阵 P_init (2×2)
            enable_gating : 是否启用休眠态门控逻辑
            innov_clip : 新息饱和阈值 γ (标准差倍数, ≤0 禁用)
            T_gap_min  : 最小沉默阈值 (秒), 防止活跃期误触发休眠
        """
        self.dt = dt
        self.R0 = R0
        self.phi = phi
        self.q = q
        self.alpha_g = alpha_g
        self.eps = eps
        self.P_init = P_init.copy()
        self.enable_gating = enable_gating
        self.innov_clip = innov_clip
        self._T_gap_min = T_gap_min

        # 转移矩阵 F (Eq. transition, L197–L202)
        self.F = np.array([[1.0, dt],
                           [0.0, 1.0]])

        # 观测矩阵 H (Eq. observation, L208–L210)
        self.H = np.array([[1.0, 0.0]])

        # 过程噪声结构矩阵 G —— 恒加速度白噪声模型 (Eq. Q_model, L270–L274)
        self.G = np.array([
            [dt**3 / 3.0, dt**2 / 2.0],
            [dt**2 / 2.0, dt],
        ])

        self.reset()

    # ----------------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------------

    def reset(self):
        """重置滤波器到初始状态。"""
        self.x = np.zeros(2)           # 状态 x̂_{0|0} = 0
        self.P = self.P_init.copy()    # 协方差 P_{0|0} = P_init
        self.t_last = 0.0              # 上次事件时间
        self.dormant = False           # 休眠态标志
        self._just_woke = True         # 首步视同唤醒 (P=P_init, x=0)
        self._lam_anchor = self.eps    # 锚定强度: 最后有事件时刻的后验估计

        # 内部状态历史 (用于后续绘图与分析)
        self.history = dict(
            lambda_hat=[],    # 后验强度估计
            lambda_dot=[],    # 后验速率估计
            K_gain=[],        # 位置分量 Kalman 增益
            tau_eff=[],       # 有效带宽 τ_eff
            dormant_flag=[],  # 休眠态标志
        )

    def step(self, k, dN_k):
        """
        执行单步 predict-update 循环。

        参数:
            k     : 当前时间步索引 (1-indexed)
            dN_k  : 当前 bin 内的事件计数 ΔN_k

        返回:
            后验强度估计 λ̂_{k|k}
        """
        z_k = dN_k / self.dt   # 原始测量值 z_k = ΔN_k / Δt  (Eq. measurement)

        # ---- 事件到达与唤醒逻辑 (L534–L541) ----
        if dN_k > 0:
            self.t_last = k * self.dt
            if self.dormant:
                self.dormant = False
                self._just_woke = True        # 标记唤醒态
                self.P = self.P_init.copy()   # 协方差重膨胀 (L539)
            # 锚定强度将在本步 update 后更新 (见下方)

        # ---- 休眠态处理 (L543–L546) ----
        if self.dormant:
            self.x = np.zeros(2)
            self._record(0.0, 0.0, 0.0, np.inf, True)
            return 0.0

        # ---- 间隙检测: 仅活跃态 (L548–L554) ----
        # 使用锚定强度 (最后事件时刻的后验估计) 计算阈值,
        # 防止连续衰减导致 T_gap 不断变大而永远不触发门控。
        if self.enable_gating:
            T_silent = k * self.dt - self.t_last
            T_gap = -np.log(self.alpha_g) / self._lam_anchor  # (Eq. gap_threshold)
            T_gap = max(T_gap, self._T_gap_min)               # 最小沉默阈值
            if T_silent > T_gap:
                self.dormant = True
                self.x = np.zeros(2)
                self._record(0.0, 0.0, 0.0, np.inf, True)
                return 0.0

        # ---- 内生噪声参数 (L556–L558) ----
        lam_prev = max(self.x[0], self.eps)
        # 唤醒态 R 保护: 刚唤醒时 λ̂=0, 用 z_k 作为 R 的异方差参考
        if self._just_woke and z_k > 0:
            lam_for_R = z_k
            self._just_woke = False
        else:
            lam_for_R = lam_prev
            self._just_woke = False
        R_k = self.R0 + self.phi * lam_for_R / self.dt       # (Eq. R_model)
        Q_k = self.q * max(lam_prev**2, self.eps) * self.G  # (Eq. Q_model)

        # ---- 预测步 (L559–L561) ----
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + Q_k

        # ---- 更新步 (L562–L566) ----
        S_k = (self.H @ P_pred @ self.H.T).item() + R_k   # 新息方差 (标量)
        K_k = (P_pred @ self.H.T) / S_k                  # Kalman 增益 (2×1)
        innov = z_k - (self.H @ x_pred).item()

        # 新息饱和: 截断超过 γ 倍标准差的极端新息 (robust KF)
        if self.innov_clip > 0 and S_k > 0:
            innov_std = innov / np.sqrt(S_k)
            if abs(innov_std) > self.innov_clip:
                innov = self.innov_clip * np.sqrt(S_k) * np.sign(innov)

        self.x = x_pred + K_k.flatten() * innov
        # Joseph 形式协方差更新 (数值更稳定, 保持 PSD)
        I_KH = np.eye(2) - K_k @ self.H
        self.P = I_KH @ P_pred @ I_KH.T + R_k * (K_k @ K_k.T)

        # ---- 非负投影 (L567–L568, Eq. projection) ----
        self.x[0] = max(self.x[0], 0.0)

        # ---- 速度约束: 防止 CV 模型速度状态在稀疏区过冲 ----
        # |λ̇| ≤ C·λ̂，限制单步强度变化率在当前估计的合理比例内
        lam_now = max(self.x[0], self.eps)
        max_rate = 0.1 * lam_now
        self.x[1] = np.clip(self.x[1], -max_rate, max_rate)

        # ---- 更新锚定强度 (在有事件到达时锁定后验值) ----
        if dN_k > 0:
            self._lam_anchor = max(self.x[0], self.eps)

        # ---- 记录有效带宽 ----
        K_pos = float(K_k[0, 0])   # 位置分量增益
        if 0.0 < K_pos < 1.0:
            tau = -self.dt / np.log(1.0 - K_pos)   # (Eq. tau_eff)
        else:
            tau = self.dt / max(K_pos, self.eps)    # 近似 (Eq. tau_approx)

        self._record(self.x[0], self.x[1], K_pos, tau, False)
        return self.x[0]

    def process_sequence(self, counts):
        """
        处理完整的 bin 计数序列。

        参数:
            counts : 长度 K 的数组, 各 bin 的事件计数

        返回:
            长度 K 的强度估计数组
        """
        self.reset()
        K = len(counts)
        estimates = np.zeros(K)
        for k in range(K):
            estimates[k] = self.step(k + 1, counts[k])
        return estimates

    # ----------------------------------------------------------------
    # 内部工具
    # ----------------------------------------------------------------

    def _record(self, lam, lam_dot, K, tau, dormant):
        """将当前步的内部变量追加到历史记录。"""
        self.history['lambda_hat'].append(lam)
        self.history['lambda_dot'].append(lam_dot)
        self.history['K_gain'].append(K)
        self.history['tau_eff'].append(tau)
        self.history['dormant_flag'].append(dormant)
