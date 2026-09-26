#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

Matrix ReadMatrixFromFile(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open file " + path);
    }

    Matrix matrix;
    std::string line;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        double number;
        std::vector<double> row_tokens;

        while (ss >> number) {
            row_tokens.emplace_back(number);
        }

        if (!row_tokens.empty()) {
            matrix.emplace_back(row_tokens);
        }
    }

    in.close();
    return matrix;
}

Matrix MultiplyMatrices(const Matrix& lhs, const Matrix& rhs) {
    Matrix result(lhs.size(), std::vector<double>(rhs[0].size(), 0));

    for (size_t i = 0; i < lhs.size(); ++i) {
        for (size_t j = 0; j < rhs[0].size(); ++j) {
            for (size_t k = 0; k < rhs.size(); ++k) {
                result[i][j] += lhs[i][k] * rhs[k][j];
            }
        }
    }

    return result;
}

double VectorNormFrom(const Matrix& matrix, size_t column, size_t start_row) {
    double sum = 0;

    for (size_t i = start_row; i < matrix.size(); ++i) {
        sum += matrix[i][column] * matrix[i][column];
    }

    return std::sqrt(sum);
}

std::pair<Matrix, Matrix> QRDecomposition(const Matrix& matrix) {
    size_t n = matrix.size();

    Matrix q = Matrix(n, std::vector<double>(n, 0));
    for (size_t i = 0; i < n; ++i) {
        q[i][i] = 1;
    }

    Matrix r = matrix;

    for (size_t k = 0; k + 1 < n; ++k) {
        double norm = VectorNormFrom(r, k, k);
        if (norm == 0) {
            continue;
        }

        Matrix v(n, std::vector<double>(1, 0));
        double sign = r[k][k] >= 0 ? 1 : -1;
        v[k][0] = r[k][k] + sign * norm;

        for (size_t i = k + 1; i < n; ++i) {
            v[i][0] = r[i][k];
        }

        double v_norm_squared = 0;
        for (size_t i = k; i < n; ++i) {
            v_norm_squared += v[i][0] * v[i][0];
        }
        if (v_norm_squared == 0) {
            continue;
        }

        Matrix h(n, std::vector<double>(n, 0));
        for (size_t p = 0; p < n; ++p) {
            h[p][p] = 1;
        }

        for (size_t i = k; i < n; ++i) {
            for (size_t j = k; j < n; ++j) {
                h[i][j] -= 2 * v[i][0] * v[j][0] / v_norm_squared;
            }
        }

        r = MultiplyMatrices(h, r);
        q = MultiplyMatrices(q, h);
    }

    return {q, r};
}

std::pair<Complex, Complex> Eigenvalues2x2(double a, double b, double c, double d) {
    double trace = a + d;
    double determinant = a * d - b * c;
    double discriminant = trace * trace - 4 * determinant;

    if (discriminant >= 0) {
        double root = std::sqrt(discriminant);
        return {
            Complex((trace + root) / 2, 0),
            Complex((trace - root) / 2, 0)
        };
    }

    return {
        Complex(trace / 2, std::sqrt(-discriminant) / 2),
        Complex(trace / 2, -std::sqrt(-discriminant) / 2)
    };
}

ComplexVector ExtractEigenvalues(const Matrix& matrix, double eps) {
    ComplexVector eigenvalues;

    for (size_t i = 0; i < matrix.size();) {
        if (i + 1 < matrix.size() && std::abs(matrix[i + 1][i]) > eps) {
            auto [first, second] = Eigenvalues2x2(
                matrix[i][i],
                matrix[i][i + 1],
                matrix[i + 1][i],
                matrix[i + 1][i + 1]
            );
            eigenvalues.emplace_back(first);
            eigenvalues.emplace_back(second);
            i += 2;
        } else {
            eigenvalues.emplace_back(matrix[i][i], 0);
            ++i;
        }
    }

    return eigenvalues;
}

bool IsQRIterationFinished(const Matrix& current, const Matrix& previous, double eps) {
    for (size_t i = 0; i + 1 < current.size(); ++i) {
        if (VectorNormFrom(current, i, i + 1) <= eps) {
            continue;
        }

        auto [cur_first, cur_second] = Eigenvalues2x2(
            current[i][i],
            current[i][i + 1],
            current[i + 1][i],
            current[i + 1][i + 1]
        );
        auto [prev_first, prev_second] = Eigenvalues2x2(
            previous[i][i],
            previous[i][i + 1],
            previous[i + 1][i],
            previous[i + 1][i + 1]
        );

        if (std::abs(cur_first - prev_first) > eps ||
            std::abs(cur_second - prev_second) > eps) {
            return false;
        }

        ++i;
    }

    return true;
}

ComplexVector FindEigenvaluesQR(const Matrix& matrix, double eps) {
    auto [q, r] = QRDecomposition(matrix);
    Matrix previous = matrix;
    Matrix current = MultiplyMatrices(r, q);
    constexpr size_t kMaxIterations = 1000;

    for (size_t i = 1; i < kMaxIterations; ++i) {
        if (IsQRIterationFinished(current, previous, eps)) {
            break;
        }

        previous = current;
        auto [next_q, next_r] = QRDecomposition(current);
        current = MultiplyMatrices(next_r, next_q);
    }

    return ExtractEigenvalues(current, eps);
}

void PrintMatrix(const Matrix& matrix) {
    std::cout << std::fixed << std::setprecision(6);

    for (const auto& row : matrix) {
        for (double value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

void PrintComplexVector(const ComplexVector& vector) {
    std::cout << std::fixed << std::setprecision(6);

    for (const Complex& value : vector) {
        std::cout << value.real();
        if (std::abs(value.imag()) > std::numeric_limits<double>::epsilon()) {
            std::cout << (value.imag() >= 0 ? "+" : "-")
                      << std::abs(value.imag()) << "i";
        }
        std::cout << std::endl;
    }
}

int main() {
    constexpr double kEps = 0.01;

    Matrix matrix = ReadMatrixFromFile("matrix.txt");
    auto [q, r] = QRDecomposition(matrix);
    ComplexVector eigenvalues = FindEigenvaluesQR(matrix, kEps);

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "1. Initial matrix: " << std::endl;
    PrintMatrix(matrix);

    std::cout << "2. Q and R matrices: " << std::endl;
    PrintMatrix(q);
    std::cout << std::endl;
    PrintMatrix(r);

    std::cout << "3. Result of product Q and R: " << std::endl;
    PrintMatrix(MultiplyMatrices(q, r));

    std::cout << "4. Eigenvalues: " << std::endl;
    PrintComplexVector(eigenvalues);

    std::cout << "5. Example with complex eigenvalues: " << std::endl;
    Matrix complex_example = {
        {1, 3, 1},
        {1, 1, 4},
        {4, 3, 1}
    };
    ComplexVector complex_eigenvalues = FindEigenvaluesQR(complex_example, kEps);

    std::cout << "Initial matrix:" << std::endl;
    PrintMatrix(complex_example);
    std::cout << "Eigenvalues:" << std::endl;
    PrintComplexVector(complex_eigenvalues);

    return 0;
}