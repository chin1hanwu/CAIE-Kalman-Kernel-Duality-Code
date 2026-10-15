# CAIE Kalman-Kernel Duality Code

[![CI](https://github.com/chin1hanwu/CAIE-Kalman-Kernel-Duality-Code/actions/workflows/ci.yml/badge.svg)](https://github.com/chin1hanwu/CAIE-Kalman-Kernel-Duality-Code/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

用于事件流强度估计的 Python 研究代码，包含标量与二维自适应 Observer、EWMA、标准 Kalman 滤波和离线 Oracle KDE 基线，以及合成事件生成和评估工具。

## 安装与快速运行

需要 Python 3.10–3.13。在仓库根目录执行：

```bash
python -m venv .venv
# Git Bash / Linux / macOS
source .venv/Scripts/activate  # Windows Git Bash
# source .venv/bin/activate  # Linux / macOS
python -m pip install -r requirements.txt
python examples/quickstart.py
python -m unittest discover -s tests -v
```

快速示例使用固定随机种子生成事件计数，运行两种 Observer，并打印 RMSE。CI 在 Linux 的 Python 3.10、3.12、3.13 和 Windows 的 Python 3.13 上检查核心功能；神经数据接口在独立任务中安装可选依赖后检查。

## 代码入口

| 文件 | 功能 |
| --- | --- |
| `src/observer.py` | `AdaptiveObserver1D` 与二维 `AdaptiveObserver`，包含休眠门控 |
| `src/baselines.py` | EWMA、标准 Kalman、倒数估计和离线 Oracle KDE |
| `src/data_gen.py` | NHPP、Hawkes、Gamma 更新过程与事件分箱 |
| `src/config.py` | 时间步长、随机种子、模型参数与实验配置 |
| `src/metrics.py`、`src/metrics_realworld.py` | 误差、滞后、预测对数似然与时间重缩放指标 |
| `src/exp1_reciprocal_divergence.py` | Gamma 更新过程中的倒数估计器偏差 |
| `src/exp2_bandwidth_adaptation.py` | 带宽缩放与非平稳跟踪 |
| `src/exp3_hawkes_avalanche.py` | 自激事件强度跟踪 |
| `src/exp4_leakage_ablation.py` | 休眠门控消融 |
| `src/app_finance.py`、`src/app_neuro.py` | 使用自行准备的本地数据运行分析 |
| `src/app_complexity.py` | 合成 Poisson 事件上的运行时间基准 |

运行全部合成实验：

```bash
python src/run_section6.py
```

各实验也可单独执行，例如 `python src/exp3_hawkes_avalanche.py`。图形写入 `figures/`。无图形界面的环境可设置 `MPLBACKEND=Agg`。完整配置的倒数估计实验包含多次重复；Oracle KDE 与大规模运行时间基准可能耗时较长。

## 使用 Observer

`examples/quickstart.py` 给出了完整示例。构造参数见 `src/config.py`。输入 `counts[k]` 为宽度 `DT` 内的事件数量，输出强度单位为事件数/秒。`process_sequence(counts)` 每次都会重置状态；在线连续处理使用 `step(k, count)`，其中 `k` 从 1 开始依次增加。

历史量通过 `observer.history` 访问，包括 `lambda_hat`、`K_gain`、`tau_eff` 和 `dormant_flag`。Oracle KDE 使用完整事件序列作为离线参照。预测对数似然应使用观测当前分箱之前的预测强度；合成跟踪误差可使用更新后的估计。

## 可选外部数据

本仓库不附带数据，也不自动下载数据。请按数据提供方的许可和使用条件自行获取。

金融分析使用 [Binance Public Data](https://github.com/binance/binance-public-data) 的 BTCUSDT aggTrades，需准备以下两个无表头的 CSV：

```text
data/finance/BTCUSDT-aggTrades-2025-10-10.csv
data/finance/BTCUSDT-aggTrades-2025-10-11.csv
```

每行至少包含六列，列索引从 0 开始：价格为第 1 列、数量为第 2 列、时间戳为第 5 列。加载器将 Binance 现货 2025 年起使用的微秒时间戳转换为秒，时间原点为 UTC 2025-10-10。运行 `python src/app_finance.py`。

神经数据分析使用 [DANDI 000128 / MC_Maze](https://dandiarchive.org/dandiset/000128)。先执行 `python -m pip install -r requirements-neuro.txt`，再准备：

```text
data/neuro/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb
```

运行 `python src/app_neuro.py`。外部数据的端到端结果需在取得相应数据后验证；CI 运行合成测试和接口检查。运行时间结果取决于设备和配置。

## 许可证

代码以 [Apache License 2.0](LICENSE) 发布，版权信息见 [NOTICE](NOTICE)。第三方依赖与外部数据遵循各自的许可证。
