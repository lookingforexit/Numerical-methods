#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
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

double calcDeterminant(const Matrix& matrix) {
    if (matrix.empty()) {
        throw std::runtime_error("Matrix is empty");
    }

    if (matrix.size() == 1) {
        return matrix[0][0];
    }

    double determinant = 0;

    for (size_t i = 0 ; i < matrix.size() ; ++i) {
        Matrix smallMatrix;
        for (size_t j = 1; j < matrix.size(); ++j) {
            for (size_t k = 0; k < matrix.size(); ++k) {
                if (k != i) {
                    smallMatrix.emplace_back(matrix[j][k]);
                }
            }
        }

        if (i & 1) {
            determinant += -matrix[0][i] * calcDeterminant(smallMatrix);
        } else {
            determinant += matrix[0][i] * calcDeterminant(smallMatrix);
        }
    }

    return determinant;
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

std::tuple<Matrix, Matrix, Matrix, size_t> LUDecomposition(Matrix matrix) {
    if (matrix.empty()) {
        throw std::invalid_argument("Matrix is empty");
    }

    Matrix permutationMatrix(matrix.size(), std::vector<double>(matrix.size(), 0));
    for (size_t i = 0; i < permutationMatrix.size(); ++i) {
        permutationMatrix[i][i] = 1;
    }

    size_t countOfPermutations = 0;

    for (size_t col = 0; col < matrix.size(); ++col) {
        double maxInCol = -std::numeric_limits<double>::max();
        size_t posMaxInCol = 0;

        for (size_t row = col; row < matrix.size(); ++row) {
            if (std::abs(matrix[row][col]) > maxInCol) {
                maxInCol = std::abs(matrix[row][col]);
                posMaxInCol = row;
            }
        }

        if (std::abs(maxInCol) < std::numeric_limits<double>::epsilon()) {
            throw std::invalid_argument("Matrix is singular");
        }

        if (posMaxInCol != col) {
            auto temp = matrix[col];
            matrix[col] = matrix[posMaxInCol];
            matrix[posMaxInCol] = temp;

            temp = permutationMatrix[col];
            permutationMatrix[col] = permutationMatrix[posMaxInCol];
            permutationMatrix[posMaxInCol] = temp;
            ++countOfPermutations;
        }

        for (size_t i = col + 1; i < matrix.size(); ++i) {
            matrix[i][col] = matrix[i][col] / matrix[col][col];
            for (size_t j = col + 1; j < matrix.size(); ++j) {
                matrix[i][j] -= matrix[i][col] * matrix[col][j];
            }
        }
    }

    Matrix upperMatrix(matrix.size(), std::vector<double>(matrix.size(), 0));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = i; j < matrix.size(); ++j) {
            upperMatrix[i][j] = matrix[i][j];
        }
    }

    Matrix lowerMatrix(matrix.size(), std::vector<double>(matrix.size(), 0));
    for (size_t i = 0; i < matrix.size(); ++i) {
        lowerMatrix[i][i] = 1;
        for (size_t j = 0; j < i; ++j) {
            lowerMatrix[i][j] = matrix[i][j];
        }
    }

    return {lowerMatrix, upperMatrix, permutationMatrix, countOfPermutations};
}

Matrix LUSolve(const Matrix& l, const Matrix& u, const Matrix& p, const Matrix& b) {
    Matrix z(l.size(), std::vector<double>(1, 0));
    for (size_t i = 0; i < z.size(); ++i) {
        size_t target = 0;
        for (size_t j = 0; j < p.size(); ++j) {
            if (std::abs(p[i][j] - 1) < std::numeric_limits<double>::epsilon()) {
                target = j;
                break;
            }
        }

        double sum = 0;
        for (size_t j = 0; j < i; ++j) {
            sum += l[i][j] * z[j][0];
        }
        z[i][0] = b[target][0] - sum;
    }

    Matrix x(l.size(), std::vector<double>(1, 0));
    for (size_t i = 0; i < x.size(); ++i) {
        size_t k = x.size() - i - 1;

        double sum = 0;
        for (size_t j = k + 1; j < x.size(); ++j) {
            sum += u[k][j] * x[j][0];
        }
        x[k][0] = (z[k][0] - sum) / u[k][k];
    }

    return x;
}

double determinantLU(const Matrix& u, size_t countOfPermutations) {
    double determinant = 1;
    for (size_t i = 0; i < u.size(); ++i) {
        determinant *= u[i][i];
    }

    if (countOfPermutations & 1) {
        determinant = -determinant;
    }

    return determinant;
}

Matrix calculateInverse(const Matrix& l, const Matrix& u, const Matrix& p) {
    Matrix inverse(l.size(), std::vector<double>(l.size(), 0));
    Matrix temp(l.size(), std::vector<double>(1, 0));

    for (size_t i = 0; i < l.size(); ++i) {
        for (size_t j = 0; j < l.size(); ++j) {
            temp[j][0] = (i == j) ? 1 : 0;
        }

        Matrix x = LUSolve(l, u, p, temp);

        for (size_t j = 0; j < l.size(); ++j) {
            inverse[j][i] = x[j][0];
        }
    }

    return inverse;
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

int main(int argc, char** argv) {
    Matrix fullMatrix = readMatrixFromFile("matrix.txt");

    Matrix matrix(fullMatrix.size(), std::vector<double>(fullMatrix.size(), 0));
    for (size_t i = 0; i < fullMatrix.size(); ++i) {
        for (size_t j = 0; j < fullMatrix.size(); ++j) {
            matrix[i][j] = fullMatrix[i][j];
        }
    }

    Matrix b(fullMatrix.size(), std::vector<double>(1, 0));
    for (size_t i = 0; i < fullMatrix.size(); ++i) {
        b[i][0] = fullMatrix[i][fullMatrix[0].size() - 1];
    }

    auto [l, u, p, permsCount] = LUDecomposition(matrix);

    std::cout << "1. L and U matrices:" << std::endl;
    printMatrix(l);
    std::cout << std::endl;
    printMatrix(u);
    std::cout << std::endl;

    std::cout << "2. Result of product L and U:" << std::endl;
    printMatrix(multiplyMatrices(l, u));
    std::cout << std::endl;

    std::cout << "3. Solution of Ax = b:" << std::endl;
    printMatrix(LUSolve(l, u, p, b));
    std::cout << std::endl;

    std::cout << "4. Inverse of A:" << std::endl;
    printMatrix(calculateInverse(l, u, p));
    std::cout << std::endl;

    std::cout << "5. Determinant of A:" << std::endl;
    std::cout << std::fixed << std::setprecision(6) << determinantLU(u, permsCount) << std::endl;
    std::cout << std::endl;

    std::cout << "6. Result of product A and A^(-1):" << std::endl;
    printMatrix(multiplyMatrices(matrix, calculateInverse(l, u, p)));
    std::cout << std::endl;

    std::cout << "7. Result of products P and A, L and U:" << std::endl;
    printMatrix(multiplyMatrices(p, matrix));
    std::cout << std::endl;
    printMatrix(multiplyMatrices(l, u));
    std::cout << std::endl;

    return 0;
}