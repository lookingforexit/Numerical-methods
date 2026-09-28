from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


OUTPUT_PATH = Path(__file__).with_name("graphic.png")


def main():
    x = np.linspace(0.2, 0.45, 500)

    first_equation_x = x[(3.0 * x >= -1.0) & (3.0 * x <= 1.0)]
    first_equation_y = np.arccos(3.0 * first_equation_x)
    second_equation_y = np.exp(x) / 3.0

    plt.figure(figsize=(11, 7))
    plt.plot(first_equation_x, first_equation_y, label="3x - cos(y) = 0")
    plt.plot(x, second_equation_y, label="3y - e^x = 0")

    plt.xlim(0.2, 0.45)
    plt.ylim(0.35, 0.6)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
