from math import log, pow


def f(x: float) -> float:
    return pow(4.0, x) - 5.0 * x - 2.0


def df(x: float) -> float:
    return log(4.0) * pow(4.0, x) - 5.0


def phi(x: float) -> float:
    return log(5.0 * x + 2.0) / log(4.0)


def dphi(x: float) -> float:
    return 5.0 / ((5.0 * x + 2.0) * log(4.0))


def simple_iteration(x: float, eps: float) -> tuple[float, int]:
    for iteration in range(1, 1001):
        next_x = phi(x)

        if abs(next_x - x) <= eps and abs(f(next_x)) <= eps:
            return next_x, iteration

        x = next_x

    raise RuntimeError("iters for simple iterations is too big")


def newton(x: float, eps: float) -> tuple[float, int]:
    for iteration in range(1, 1001):
        next_x = x - f(x) / df(x)

        if abs(next_x - x) <= eps and abs(f(next_x)) <= eps:
            return next_x, iteration

        x = next_x

    raise RuntimeError("iters for newton is too big")


def read_eps() -> float:
    eps = float(input("Eps: "))
    if eps <= 0.0:
        raise ValueError
    return eps


def main():
    print("4^x - 5x - 2 = 0")
    eps = read_eps()

    x0 = 1.65
    print("phi(x) = log_4(5x + 2)")
    print(f"|phi'(x0)| = {abs(dphi(x0)):.6f} < 1\n")

    iteration_root, iteration_count = simple_iteration(x0, eps)
    newton_root, newton_count = newton(x0, eps)

    print("simple iters:")
    print(f"x = {iteration_root:.6f}")
    print(f"f(x) = {f(iteration_root):.6f}")
    print(f"iters = {iteration_count}\n")

    print("newton:")
    print(f"x = {newton_root:.6f}")
    print(f"f(x) = {f(newton_root):.6f}")
    print(f"iters = {newton_count}")


if __name__ == "__main__":
    main()