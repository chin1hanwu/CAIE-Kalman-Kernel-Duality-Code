# SPDX-License-Identifier: Apache-2.0
"""无需外部数据的强度估计示例。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import DT, R0, PHI, Q_SCALE, ALPHA_G, EPS, P_INIT
from observer import AdaptiveObserver, AdaptiveObserver1D


def main():
    rng = np.random.default_rng(42)
    true_rate = np.repeat([5.0, 20.0, 5.0], 200)
    counts = rng.poisson(true_rate * DT)
    for observer_type in (AdaptiveObserver1D, AdaptiveObserver):
        observer = observer_type(DT, R0, PHI, Q_SCALE, ALPHA_G, EPS, P_INIT)
        estimate = observer.process_sequence(counts)
        if not np.all(np.isfinite(estimate)) or np.any(estimate < 0):
            raise RuntimeError("强度估计出现无效数值")
        error = np.sqrt(np.mean((estimate - true_rate) ** 2))
        print(f"{observer_type.__name__}: 分箱数={len(counts)}, RMSE={error:.3f}")


if __name__ == "__main__":
    main()
