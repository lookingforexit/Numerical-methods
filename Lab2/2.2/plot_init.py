from pathlib import Path
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


OUTPUT_PATH = Path(__file__).with_name("graphic.png")


def f1(x, y):
    return 3.0 * x - np.cos(y)


def f2(x, y):
    return 3.0 * y - np.exp(x)


def main():
    x = np.linspace(-0.2, 1.2, 500)
    y_from_second = np.exp(x) / 3.0

    mask = (3.0 * x >= -1.0) & (3.0 * x <= 1.0)
    y_from_first = np.arccos(3.0 * x[mask])
    x0 = 0.30
    y0 = 0.45

    plt.figure(figsize=(11, 7))
    plt.plot(x[mask], y_from_first, label="3x - cos(y) = 0", linewidth=2.2)
    plt.plot(x, y_from_second, label="3y - e^x = 0", linewidth=2.2)
    plt.scatter([x0], [y0], color="#d62828", zorder=5)

    plt.annotate(
        "x0 = 0.30, y0 = 0.45",
        xy=(x0, y0),
        xytext=(0.46, 0.30),
        arrowprops={"arrowstyle": "->", "linewidth": 1.2},
        fontsize=11,
    )

    plt.xlim(0.0, 0.8)
    plt.ylim(0.2, 0.8)
    plt.grid(True, linestyle=":", linewidth=0.8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
