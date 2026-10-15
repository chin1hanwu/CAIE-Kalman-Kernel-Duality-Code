# SPDX-License-Identifier: Apache-2.0
"""运行神经数据、金融数据及运行时间基准；须先准备本地数据。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time


def main():
    t_start = time.perf_counter()

    print('*' * 60)
    print('*  Section 7: Cross-Domain Applications')
    print('*' * 60)

    # ---- 7.2 Application I: Computational Neuroscience ----
    print('\n' + '=' * 60)
    print('  7.2 Application I: Computational Neuroscience')
    print('=' * 60)
    from app_neuro import run as run_neuro
    neuro_results = run_neuro(save=True)

    # ---- 7.3 Application II: High-Frequency Finance ----
    print('\n' + '=' * 60)
    print('  7.3 Application II: High-Frequency Finance')
    print('=' * 60)
    from app_finance import run as run_finance
    finance_results = run_finance(save=True)

    # ---- 7.4 Computational Profiling ----
    print('\n' + '=' * 60)
    print('  7.4 Computational Profiling')
    print('=' * 60)
    from app_complexity import run as run_complexity
    complexity_results = run_complexity(save=True)

    elapsed = time.perf_counter() - t_start
    print('\n' + '=' * 60)
    print('  Section 7 全部完成: 总耗时 {:.1f} s ({:.1f} min)'.format(
        elapsed, elapsed / 60))
    print('=' * 60)

    # ---- 汇总输出文件 ----
    from config import FIGURE_DIR
    print('\n生成的图表:')
    for fig_file in sorted(FIGURE_DIR.glob('*.pdf')):
        print('  - {}'.format(fig_file))


if __name__ == '__main__':
    main()
