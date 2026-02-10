"""
전체 실행 진입점
=================
Cross-sectional table + 주요 figure 생성.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ldi import params as P
from ldi.compare import (
    cross_sectional_table,
    plot_cross_sectional,
    plot_time_series,
    plot_eps_sensitivity,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)


def main():
    P.print_params()
    print()

    print("=== Cross-Sectional Table ===")
    cross_sectional_table()
    print()

    plot_cross_sectional(save_path=os.path.join(OUT, "cross_sectional.png"))
    plot_time_series(save_path=os.path.join(OUT, "time_series.png"))
    plot_eps_sensitivity(save_path=os.path.join(OUT, "eps_sensitivity.png"))

    print("\nAll figures generated in outputs/")


if __name__ == "__main__":
    main()
