# SPDX-License-Identifier: Apache-2.0
"""依次运行神经数据与金融数据分析；须先准备本地数据。"""

def main():
    from app_neuro import run as run_neuro
    from app_finance import run as run_finance
    run_neuro(save=True)
    run_finance(save=True)

if __name__ == "__main__":
    main()
