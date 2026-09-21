#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

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

Matrix Transpose(const Matrix& A) {
    Matrix T(A[0].size(), std::vector<double>(A.size()));
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

Matrix MultiplyMatrices(const Matrix& lhs, const Matrix& rhs) {
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

Matrix RotationMatrix(const Matrix& matrix, size_t i, size_t j) {
    Matrix rotation_matrix(matrix.size(), std::vector<double>(matrix[0].size(), 0));
    for (size_t k = 0; k < matrix.size(); ++k) {
        rotation_matrix[k][k] = 1;
    }

    double phi = 0;

    if (std::abs(matrix[i][i] - matrix[j][j]) < std::numeric_limits<double>::epsilon()) {
        constexpr double kPi = 3.141592;
        phi = kPi / 4;
    } else {
        phi = 0.5 * std::atan(2 * matrix[i][j] / (matrix[i][i] - matrix[j][j]));
    }

    double cos_phi = std::cos(phi);
    double sin_phi = std::sin(phi);

    rotation_matrix[i][i] = cos_phi;
    rotation_matrix[j][j] = cos_phi;
    rotation_matrix[i][j] = -sin_phi;
    rotation_matrix[j][i] = sin_phi;

    return rotation_matrix;
}

double OffDiagonalNorm(const Matrix& matrix) {
    double sum = 0;

    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = i + 1; j < matrix[i].size(); ++j) {
            sum += matrix[i][j] * matrix[i][j];
        }
    }

    return std::sqrt(sum);
}

std::tuple<Matrix, Matrix> JacobiRotationMethod(Matrix matrix, const double eps) {
    Matrix eigenvectors(matrix.size(), std::vector<double>(matrix[0].size(), 0));
    for (size_t i = 0; i < matrix.size(); ++i) {
        eigenvectors[i][i] = 1;
    }

    while (OffDiagonalNorm(matrix) > eps) {
        double max_value = 0;
        size_t max_i = 0;
        size_t max_j = 0;

        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = i + 1; j < matrix[i].size(); ++j) {
                if (std::abs(matrix[i][j]) > max_value) {
                    max_value = std::abs(matrix[i][j]);
                    max_i = i;
                    max_j = j;
                }
            }
        }

        Matrix rotation_matrix = RotationMatrix(matrix, max_i, max_j);
        matrix = MultiplyMatrices(MultiplyMatrices(Transpose(rotation_matrix), matrix), rotation_matrix);
        eigenvectors = MultiplyMatrices(eigenvectors, rotation_matrix);
    }

    Matrix eigenvalues(matrix.size(), std::vector<double>(1, 0));
    for (size_t i = 0; i < matrix.size(); ++i) {
        eigenvalues[i][0] = matrix[i][i];
    }

    return {eigenvalues, eigenvectors};
}

Matrix DiagonalMatrix(const Matrix& eigenvalues) {
    Matrix result(eigenvalues.size(), std::vector<double>(eigenvalues.size(), 0));
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        result[i][i] = eigenvalues[i][0];
    }
    return result;
}

void PrintMatrix(const Matrix& matrix) {
    std::cout << std::fixed << std::setprecision(6);

    for (const auto & i : matrix) {
        for (double j : i) {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    constexpr double kEps = 0.01;
    Matrix matrix = ReadMatrixFromFile("matrix.txt");

    auto [eigenvalues, eigenvectors] = JacobiRotationMethod(matrix, kEps);
    Matrix diagonal = DiagonalMatrix(eigenvalues);

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "1. Epsilon: " << std::endl;
    std::cout << kEps << std::endl;

    std::cout << "2. Eigenvalues: " << std::endl;
    PrintMatrix(eigenvalues);

    std::cout << "3. Eigenvectors: " << std::endl;
    PrintMatrix(eigenvectors);

    std::cout << "4. Results of products A*V and V*A:" << std::endl;
    std::cout << "A*V:" << std::endl;
    PrintMatrix(MultiplyMatrices(matrix, eigenvectors));
    std::cout << "V*A:" << std::endl;
    PrintMatrix(MultiplyMatrices(eigenvectors, diagonal));

    return 0;
}
