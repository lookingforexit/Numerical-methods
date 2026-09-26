from math import cos, exp, sin


def f1(x: float, y: float) -> float:
    return 3.0 * x - cos(y)


def f2(x: float, y: float) -> float:
    return 3.0 * y - exp(x)


def phi1(y: float) -> float:
    return cos(y) / 3.0


def phi2(x: float) -> float:
    return exp(x) / 3.0


def norm(x: float, y: float) -> float:
    return max(abs(x), abs(y))


def simple_iteration(x: float, y: float, eps: float) -> tuple[float, float, int, list[float]]:
    errors = []

    for iteration in range(1, 1001):
        next_x = phi1(y)
        next_y = phi2(x)
        error = norm(next_x - x, next_y - y)
        errors.append(error)

        if error <= eps and norm(f1(next_x, next_y), f2(next_x, next_y)) <= eps:
            return next_x, next_y, iteration, errors

        x = next_x
        y = next_y

    raise RuntimeError("iters for simple iterations is too big")


def newton(x: float, y: float, eps: float) -> tuple[float, float, int, list[float]]:
    errors = []

    for iteration in range(1, 1001):
        a11 = 3.0
        a12 = sin(y)
        a21 = -exp(x)
        a22 = 3.0

        det = a11 * a22 - a12 * a21
        b1 = -f1(x, y)
        b2 = -f2(x, y)

        dx = (b1 * a22 - a12 * b2) / det
        dy = (a11 * b2 - b1 * a21) / det

        x += dx
        y += dy

        error = norm(dx, dy)
        errors.append(error)

        if error <= eps and norm(f1(x, y), f2(x, y)) <= eps:
            return x, y, iteration, errors

    raise RuntimeError("iters for newton is too big")


def print_errors(title: str, errors: list[float]) -> None:
    print(title)
    print("k    error")
    for k, error in enumerate(errors, start=1):
        print(f"{k:<4} {error:.10f}")
    print()


def read_eps() -> float:
    eps = float(input("Eps: "))
    if eps <= 0.0:
        raise ValueError
    return eps


def main():
    print("3x - cos(y) = 0")
    print("3y - e^x = 0")
    eps = read_eps()

    x0 = 0.30
    y0 = 0.45
    print("\ninitial approximation:")
    print(f"x0 = {x0:.6f}, y0 = {y0:.6f}")
    print("phi1(y) = cos(y) / 3")
    print("phi2(x) = e^x / 3\n")

    iteration_x, iteration_y, iteration_count, iteration_errors = simple_iteration(x0, y0, eps)
    newton_x, newton_y, newton_count, newton_errors = newton(x0, y0, eps)

    print("simple iters:")
    print(f"x = {iteration_x:.6f}")
    print(f"y = {iteration_y:.6f}")
    print(f"f1(x, y) = {f1(iteration_x, iteration_y):.6f}")
    print(f"f2(x, y) = {f2(iteration_x, iteration_y):.6f}")
    print(f"iters = {iteration_count}\n")

    print("newton:")
    print(f"x = {newton_x:.6f}")
    print(f"y = {newton_y:.6f}")
    print(f"f1(x, y) = {f1(newton_x, newton_y):.6f}")
    print(f"f2(x, y) = {f2(newton_x, newton_y):.6f}")
    print(f"iters = {newton_count}\n")

    print_errors("simple iters errors:", iteration_errors)
    print_errors("newton errors:", newton_errors)


if __name__ == "__main__":
    main()
