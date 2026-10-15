# SPDX-License-Identifier: Apache-2.0
"""运行四组合成实验。使用方式：python src/run_section6.py。"""

import sys
import time
from pathlib import Path

# 确保 src 目录在搜索路径中
sys.path.insert(0, str(Path(__file__).parent))

from config import FIGURE_DIR


def main():
    """顺序执行全部 4 个实验。"""
    print("=" * 60)
    print("  Section 6: Synthetic Experiments — 完整执行")
    print("=" * 60)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    all_results = {}

    # ---- Experiment I ----
    print("\n" + "=" * 60)
    print("  Experiment I: Reciprocal Divergence Validation")
    print("=" * 60)
    t0 = time.time()
    from exp1_reciprocal_divergence import run as run_exp1
    all_results['exp1'] = run_exp1(save=True)
    print(f"  耗时: {time.time() - t0:.1f}s")

    # ---- Experiment II ----
    print("\n" + "=" * 60)
    print("  Experiment II: Autonomous Bandwidth Adaptation")
    print("=" * 60)
    t0 = time.time()
    from exp2_bandwidth_adaptation import run as run_exp2
    all_results['exp2'] = run_exp2(save=True)
    print(f"  耗时: {time.time() - t0:.1f}s")

    # ---- Experiment III ----
    print("\n" + "=" * 60)
    print("  Experiment III: Tracking Self-Exciting Avalanches")
    print("=" * 60)
    t0 = time.time()
    from exp3_hawkes_avalanche import run as run_exp3
    all_results['exp3'] = run_exp3(save=True)
    print(f"  耗时: {time.time() - t0:.1f}s")

    # ---- Experiment IV ----
    print("\n" + "=" * 60)
    print("  Experiment IV: Dormant-State Leakage Ablation")
    print("=" * 60)
    t0 = time.time()
    from exp4_leakage_ablation import run as run_exp4
    all_results['exp4'] = run_exp4(save=True)
    print(f"  耗时: {time.time() - t0:.1f}s")

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print("  全部实验完成。")
    print(f"  总耗时: {time.time() - t_total:.1f}s")
    print(f"  输出路径: {FIGURE_DIR}")
    print("=" * 60)

    return all_results


if __name__ == '__main__':
    main()
