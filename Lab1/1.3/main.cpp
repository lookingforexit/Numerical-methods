#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

Matrix readMatrixFromFile(const std::string& path) {
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
        std::vector<double> rowTokens;

        while (ss >> number) {
            rowTokens.emplace_back(number);
        }

        if (!rowTokens.empty()) {
            matrix.emplace_back(rowTokens);
        }
    }

    in.close();
    return matrix;
}

Matrix transpose(const Matrix& A) {
    Matrix T(A[0].size(), std::vector<double>(A.size()));
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

Matrix multiplyMatrices(const Matrix& lhs, const Matrix& rhs) {
    size_t rowsA = lhs.size();
    size_t colsA = lhs[0].size();
    size_t rowsB = rhs.size();
    size_t colsB = rhs[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Invalid size of matrices");
    }

    Matrix result(rowsA, std::vector<double>(colsB, 0));

    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            for (size_t k = 0; k < colsA; ++k) {
                result[i][j] += lhs[i][k] * rhs[k][j];
            }
        }
    }

    return result;
}

Matrix solveGauss(Matrix A, Matrix b) {
    for (size_t i = 0; i < A.size(); ++i) {
        size_t maxRow = i;
        for (size_t k = i + 1; k < A.size(); ++k) {
            if (std::abs(A[k][i]) > std::abs(A[maxRow][i])) {
                maxRow = k;
            }
        }
        std::swap(A[i], A[maxRow]);
        std::swap(b[i], b[maxRow]);

        for (size_t k = i + 1; k < A.size(); ++k) {
            double factor = A[k][i] / A[i][i];
            for (size_t j = i; j < A.size(); ++j) {
                A[k][j] -= factor * A[i][j];
            }
            b[k][0] -= factor * b[i][0];
        }
    }

    Matrix x(A.size(), std::vector<double>(1, 0));
    for (size_t i = A.size(); i-- > 0;) {
        double sum = 0;
        for (size_t j = i + 1; j < A.size(); ++j) {
            sum += A[i][j] * x[j][0];
        }
        x[i][0] = (b[i][0] - sum) / A[i][i];
    }
    return x;
}

double getQ(const Matrix& A) {
    double q = 0;
    for (size_t i = 0; i < A.size(); ++i) {
        double rowSum = 0;
        for (size_t j = 0; j < A.size(); ++j) {
            if (i != j) {
                rowSum += std::abs(A[i][j] / A[i][i]);
            }
        }
        q = std::max(q, rowSum);
    }
    return q;
}

std::pair<Matrix, size_t> solveIteration(const Matrix& A, const Matrix& b, double eps) {
    double q = getQ(A);
    Matrix x(A.size(), std::vector<double>(1, 0));
    Matrix xNew(A.size(), std::vector<double>(1, 0));
    size_t iterations = 0;

    while (true) {
        for (size_t i = 0; i < A.size(); ++i) {
            double sum = 0;
            for (size_t j = 0; j < A.size(); ++j) {
                if (i != j) {
                    sum += A[i][j] * x[j][0];
                }
            }
            xNew[i][0] = (b[i][0] - sum) / A[i][i];
        }

        ++iterations;
        double maxDiff = 0;
        for (size_t i = 0; i < A.size(); ++i) {
            maxDiff = std::max(maxDiff, std::abs(xNew[i][0] - x[i][0]));
        }

        if (q < 1 && q / (1 - q) * maxDiff <= eps) {
            break;
        }
        x = xNew;
    }

    return {xNew, iterations};
}

std::pair<Matrix, size_t> solveSeidel(const Matrix& A, const Matrix& b, double eps) {
    double q = getQ(A);
    Matrix x(A.size(), std::vector<double>(1, 0));
    Matrix xPrev(A.size(), std::vector<double>(1, 0));
    size_t iterations = 0;

    while (true) {
        xPrev = x;
        for (size_t i = 0; i < A.size(); ++i) {
            double sum1 = 0, sum2 = 0;
            for (size_t j = 0; j < i; ++j) {
                sum1 += A[i][j] * x[j][0];
            }
            for (size_t j = i + 1; j < A.size(); ++j) {
                sum2 += A[i][j] * xPrev[j][0];
            }
            x[i][0] = (b[i][0] - sum1 - sum2) / A[i][i];
        }

        ++iterations;
        double maxDiff = 0;
        for (size_t i = 0; i < A.size(); ++i) {
            maxDiff = std::max(maxDiff, std::abs(x[i][0] - xPrev[i][0]));
        }

        if (q < 1 && q / (1 - q) * maxDiff <= eps) {
            break;
        }
    }
    return {x, iterations};
}

void printMatrix(const Matrix& matrix) {
    std::cout << std::fixed << std::setprecision(6);

    for (const auto & i : matrix) {
        for (double j : i) {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) {
    Matrix fullMatrix = readMatrixFromFile("matrix.txt");
    size_t n = fullMatrix.size();

    Matrix A(n, std::vector<double>(n));
    Matrix b(n, std::vector<double>(1));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A[i][j] = fullMatrix[i][j];
        }
        b[i][0] = fullMatrix[i][n];
    }

    double eps = 0.01;

    Matrix AT = transpose(A);
    Matrix ASym = multiplyMatrices(AT, A);
    Matrix bSym = multiplyMatrices(AT, b);

    Matrix xExact = solveGauss(A, b);
    auto [xIter, itersIter] = solveIteration(ASym, bSym, eps);
    auto [xSeidel, itersSeidel] = solveSeidel(ASym, bSym, eps);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "1. Epsilon:" << std::endl;
    std::cout << eps << std::endl;

    std::cout << "2. Model solution: " << std::endl;
    printMatrix(xExact);

    std::cout << "3. Solutions: " << std::endl;
    std::cout << "Iteration method: " << std::endl;
    printMatrix(xIter);
    std::cout << "Seidel method: " << std::endl;
    printMatrix(xSeidel);

    std::cout << "4. Amount of iterations: " << std::endl;
    std::cout << "Iteration method: " << itersIter << std::endl;
    std::cout << "Seidel method: " << itersSeidel << std::endl;

    std::cout << "5. Comparing methods: " << std::endl;
    if (itersSeidel < itersIter) {
        std::cout << "Seidel faster on " << (itersIter - itersSeidel) << " iterations" << std::endl;
    } else if (itersSeidel > itersIter) {
        std::cout << "Iteration faster on " << (itersSeidel - itersIter) << " iterations" << std::endl;
    } else {
        std::cout << "Same result" << std::endl;
    }

    return 0;
}
