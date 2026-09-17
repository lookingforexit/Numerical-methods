#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using Vector = std::vector<double>;

std::tuple<Vector, Vector, Vector, Vector> readTridiagonalMatrixFromFile(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open file " + path);
    }

    Vector upperDiag;
    Vector mainDiag;
    Vector lowerDiag;
    Vector ansCol;

    std::string line;

    auto parseLine = [](const std::string& str) {
        Vector vec;
        std::istringstream iss(str);
        double val;
        while (iss >> val) {
            vec.push_back(val);
        }
        return vec;
    };

    if (std::getline(in, line)) {
        upperDiag = parseLine(line);
    }
    if (std::getline(in, line)) {
        mainDiag = parseLine(line);
    }
    if (std::getline(in, line)) {
        lowerDiag = parseLine(line);
    }
    if (std::getline(in, line)) {
        ansCol = parseLine(line);
    }

    if (mainDiag.empty()) {
        throw std::runtime_error("Matrix is empty");
    }

    if (upperDiag.size() == mainDiag.size() - 1) {
        upperDiag.push_back(0);
    }
    if (lowerDiag.size() == mainDiag.size() - 1) {
        lowerDiag.insert(lowerDiag.begin(), 0);
    }

    return {upperDiag, mainDiag, lowerDiag, ansCol};
}

std::tuple<Vector, Vector, Vector> solveTridiagonal(const Vector& upperDiag, const Vector& mainDiag, const Vector& lowerDiag, const Vector& ansCol) {
    if (ansCol.empty()) {
        throw std::runtime_error("SLAE is empty");
    }

    size_t sizeSLAE = ansCol.size();
    Vector p(sizeSLAE + 1, 0);
    Vector q(sizeSLAE + 1, 0);
    Vector x(sizeSLAE, 0);

    for (size_t i = 0; i < sizeSLAE; ++i) {
        double denominator = -mainDiag[i] - lowerDiag[i] * p[i];
        if (std::abs(denominator) < std::numeric_limits<double>::epsilon()) {
            throw std::runtime_error("Division by zero");
        }

        p[i + 1] = upperDiag[i] / denominator;
        q[i + 1] = (lowerDiag[i] * q[i] - ansCol[i]) / denominator;
    }

    double xNext = 0;
    for (size_t i = sizeSLAE; i-- > 0;) {
        x[i] = p[i + 1] * xNext + q[i + 1];
        xNext = x[i];
    }

    return {p, q, x};
}

int main(int argc, char* argv[]) {
    auto [upperDiag, mainDiag, lowerDiag, ansCol] = readTridiagonalMatrixFromFile("matrix.txt");
    auto [p, q, x] = solveTridiagonal(upperDiag,  mainDiag, lowerDiag, ansCol);

    std::cout << "1. Sweep coefficients:" << std::endl;
    for (size_t i = 0; i < p.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << "P=" << p[i] << ' ' << "Q=" << q[i] << std::endl;
    }
    std::cout << std::endl;

    std::cout << "2. Solution of SLAE:" << std::endl;
    for (size_t i = 0; i < x.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << x[i] << std::endl;
    }

    return 0;
}