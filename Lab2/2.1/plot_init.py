from pathlib import Path
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


OUTPUT_PATH = Path(__file__).with_name("graphic.png")


def f(x):
    return np.power(4.0, x) - 5.0 * x - 2.0


def main():
    x = np.linspace(0.0, 2.0, 500)
    y_exp = np.power(4.0, x)
    y_line = 5.0 * x + 2.0
    y_f = f(x)

    left = 1.6
    right = 1.7
    x0 = (left + right) / 2.0

    plt.figure(figsize=(11, 7))
    plt.plot(x, y_exp, label="y = 4^x", linewidth=2.2)
    plt.plot(x, y_line, label="y = 5x + 2", linewidth=2.2)
    plt.plot(x, y_f, label="f(x) = 4^x - 5x - 2", linewidth=2.0, linestyle="--")

    plt.axhline(0.0, color="black", linewidth=1)
    plt.axvline(0.0, color="black", linewidth=1)
    plt.axvspan(left, right, color="#ffd166", alpha=0.32, label="[1.6; 1.7]")
    plt.scatter([left, right, x0], [f(left), f(right), f(x0)], color=["#d62828", "#2a9d8f", "#1d3557"], zorder=5)

    plt.annotate(
        f"f({left}) < 0",
        xy=(left, f(left)),
        xytext=(1.18, -1.8),
        arrowprops={"arrowstyle": "->", "linewidth": 1.2},
    )
    plt.annotate(
        f"f({right}) > 0",
        xy=(right, f(right)),
        xytext=(1.72, 2.1),
        arrowprops={"arrowstyle": "->", "linewidth": 1.2},
    )
    plt.annotate(
        "x0 = 1.65",
        xy=(x0, f(x0)),
        xytext=(1.28, 0.9),
        arrowprops={"arrowstyle": "->", "linewidth": 1.2},
        fontsize=11,
    )

    plt.xlim(0.0, 2.0)
    plt.ylim(-3.4, 4.5)
    plt.grid(True, linestyle=":", linewidth=0.8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()