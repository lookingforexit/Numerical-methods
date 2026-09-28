from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


OUTPUT_PATH = Path(__file__).with_name("graphic.png")


def main():
    x = np.linspace(1.5, 1.8, 500)

    y1 = 4.0 ** x
    y2 = 5.0 * x + 2.0

    plt.figure(figsize=(11, 7))
    plt.plot(x, y1, label="y = 4^x")
    plt.plot(x, y2, label="y = 5x + 2")

    plt.xlim(1.5, 1.8)
    plt.ylim(8.5, 12.5)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
