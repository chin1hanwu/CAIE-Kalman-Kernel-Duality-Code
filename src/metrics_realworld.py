# SPDX-License-Identifier: Apache-2.0
"""
Section 7 统计诊断指标 (部分可观测情况):
  1. 一步预测对数似然  (Eq. pred_loglik, main.tex L779-L786)
  2. 时间重缩放定理检验 (Eq. time_rescaling, main.tex L788-L794)
  3. 下游线性解码 R²    (Section 7.2 Task 2)
"""

import numpy as np
from scipy import stats
from scipy.integrate import trapezoid
from scipy.special import gammaln


# ====================================================================
# 1. 一步预测对数似然 (Eq. pred_loglik)
# ====================================================================

def predictive_loglikelihood_step(lam_pred, dN, dt):
    """
    单步 Poisson 预测对数似然。

    参数:
        lam_pred : 预测强度 λ̂_{k|k-1} (标量, events/s)
        dN       : 当前 bin 内观测计数 ΔN_k
        dt       : bin 宽度 Δt

    返回:
        ℓ_k = ΔN_k·ln(λ̂·Δt) − λ̂·Δt − ln(ΔN_k!)

    注意: 计算时对 λ̂·Δt 施加下界 ε 防止 log(0)。
    """
    mu = max(lam_pred * dt, 1e-15)
    return dN * np.log(mu) - mu - _log_factorial(dN)


def cumulative_predictive_loglik(lam_pred_seq, counts, dt):
    """
    计算整个序列的累积预测对数似然 L = Σ ℓ_k。

    参数:
        lam_pred_seq : 长度 K 的预测强度序列 (events/s)
                       lam_pred_seq[k] = λ̂_{k|k-1} (使用 k-1 步的后验)
        counts       : 长度 K 的 bin 计数数组 ΔN_k
        dt           : bin 宽度

    返回:
        total : 总对数似然标量
        curve : 长度 K 的累积曲线
    """
    K = len(counts)
    curve = np.zeros(K)
    total = 0.0
    for k in range(K):
        ll = predictive_loglikelihood_step(lam_pred_seq[k], counts[k], dt)
        total += ll
        curve[k] = total
    return total, curve


def _log_factorial(n):
    """ln(n!) 使用 gammaln 实现, 支持大数。"""
    return float(gammaln(n + 1))


# ====================================================================
# 2. 时间重缩放 KS 检验 (Eq. time_rescaling)
# ====================================================================

def time_rescaling_intervals(spike_times, lam_hat_func, dt):
    """
    计算时间重缩放后的间隔 τ_i = ∫_{t_{i-1}}^{t_i} λ̂(u) du。

    参数:
        spike_times    : 长度 N 的 spike 时间戳数组 (秒)
        lam_hat_func   : 可调用对象, lam_hat_func(t) 返回 t 处的强度估计
                         或 (lam_hat_seq, dt) 元组 (bin 均匀网格上的估计值)
        dt             : bin 宽度 (用于数值积分的时间分辨率)

    返回:
        rescaled : 长度 N-1 的重缩放间隔数组
    """
    N = len(spike_times)
    if N < 2:
        return np.array([])

    rescaled = np.zeros(N - 1)
    for i in range(1, N):
        t_lo = spike_times[i - 1]
        t_hi = spike_times[i]

        # 在 [t_lo, t_hi] 上做梯形数值积分
        n_steps = max(int(np.ceil((t_hi - t_lo) / dt)), 1)
        t_grid = np.linspace(t_lo, t_hi, n_steps + 1)
        lam_vals = lam_hat_func(t_grid)
        rescaled[i - 1] = trapezoid(np.maximum(lam_vals, 0.0), t_grid)

    return rescaled


def time_rescaling_from_bins(spike_times, lam_hat_bins, dt, duration):
    """
    从 bin 网格上的强度估计计算重缩放间隔。
    使用线性插值将 bin 估计映射到连续时间。

    参数:
        spike_times  : spike 时间戳数组
        lam_hat_bins : 长度 K 的 bin 强度估计数组
        dt           : bin 宽度
        duration     : 总时长

    返回:
        rescaled : 重缩放间隔数组
    """
    # 使用分段常数 (左连续) 强度: bin k 的值覆盖 [k*dt, (k+1)*dt)
    # 避免线性插值混入相邻 bin 的值, 确保 predictable intensity 前提成立
    K = len(lam_hat_bins)
    bin_edges = np.arange(K + 1) * dt  # [0, dt, 2*dt, ..., K*dt]

    def lam_piecewise(t):
        """分段常数强度: λ(t) = lam_hat_bins[floor(t/dt)]"""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        idx = np.floor(t / dt).astype(int)
        idx = np.clip(idx, 0, K - 1)
        return lam_hat_bins[idx]

    return time_rescaling_intervals(spike_times, lam_piecewise, dt)


def ks_test_exponential(rescaled_intervals):
    """
    对重缩放间隔做 KS 检验，检验是否服从 Exp(1)。

    返回:
        ks_stat  : KS 统计量 D_N
        p_value  : 双侧 p 值
    """
    if len(rescaled_intervals) < 2:
        return np.nan, np.nan
    return stats.kstest(rescaled_intervals, 'expon', args=(0, 1))


def qq_data_exponential(rescaled_intervals):
    """
    为 Q-Q 图准备数据: 理论分位数 vs 经验分位数。

    返回:
        theoretical : 理论 Exp(1) 分位数
        empirical   : 排序后的经验值
    """
    n = len(rescaled_intervals)
    if n < 2:
        return np.array([]), np.array([])
    empirical = np.sort(rescaled_intervals)
    # 理论分位数: F^{-1}(p) = -ln(1-p), p = (i-0.5)/n
    p = (np.arange(1, n + 1) - 0.5) / n
    theoretical = -np.log(1.0 - p)
    return theoretical, empirical


# ====================================================================
# 3. 线性解码器 (Section 7.2 Task 2)
# ====================================================================

def linear_decode_r2(lam_matrix, vel_target, train_frac=0.8):
    """
    使用瞬时线性解码器从强度矩阵预测运动学 (无时滞 taps)。

    参数:
        lam_matrix : shape (T, n_units) 的强度估计矩阵
        vel_target : shape (T, n_dims) 的目标速度矩阵
        train_frac : 训练/测试划分比例

    返回:
        r2_train : 训练集 R²
        r2_test  : 测试集 R²
        vel_pred : shape (T_test, n_dims) 的预测速度
    """
    T = lam_matrix.shape[0]
    split = int(T * train_frac)

    X_train = lam_matrix[:split]
    Y_train = vel_target[:split]
    X_test = lam_matrix[split:]
    Y_test = vel_target[split:]

    # 加偏置列
    X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
    X_test_b = np.column_stack([X_test, np.ones(len(X_test))])

    # 最小二乘: W = (X^T X)^{-1} X^T Y
    # 使用 lstsq 提高数值稳定性
    W, _, _, _ = np.linalg.lstsq(X_train_b, Y_train, rcond=None)

    Y_pred_train = X_train_b @ W
    Y_pred_test = X_test_b @ W

    r2_train = _r2_score(Y_train, Y_pred_train)
    r2_test = _r2_score(Y_test, Y_pred_test)

    return r2_train, r2_test, Y_pred_test


def _r2_score(y_true, y_pred):
    """计算 R² (Macro-averaging), 支持多维。

    逐维度独立计算 R², 然后取算术平均。
    避免高方差维度主导低方差维度的贡献。
    """
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2_dims = np.where(ss_tot < 1e-15, 0.0, 1.0 - ss_res / ss_tot)
    return float(np.mean(r2_dims))
