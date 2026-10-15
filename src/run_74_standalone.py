# SPDX-License-Identifier: Apache-2.0
"""独立运行合成事件上的运行时间基准。"""

def main():
    from app_complexity import run
    run(save=True)

if __name__ == "__main__":
    main()
