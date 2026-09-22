
#include <iostream>
#include <cmath>
#include <vector>
#include "Solver.h"

int main() {
    std::vector<std::vector<double>> A = {{4.0, 1.0}, {1.0, 3.0}};
    std::vector<double> b = {1.0, 2.0};
    std::vector<double> x = {0.0, 0.0};

    SolverOptions opts = {1e-16, 1000, 0.0};
    // SolverResult info = jacobi(A, b, x, opts);
    // std::cout << info.isconverged << '\n';
    // std::cout << "Res= " << info.res << '\n';
    // std::cout << "it= " << info.it << '\n';
    // for (int i = 0; i < A.size(); ++i) {
    //     std::cout << x[i] << '\n';
    // }

    // for (int i = 0; i < info.it; ++i) {
    //     std::cout << info.history[i] << '\n';
    // }

    // return 0;

    SolverResult info_gs = GaussSeidel(A, b, x, opts);
    std::cout << info_gs.isconverged << '\n';
    std::cout << "Res= " << info_gs.res << '\n';
    std::cout << "it= " << info_gs.it << '\n';
    for (int i = 0; i < A.size(); ++i) {
        std::cout << x[i] << '\n';
    }

    for (int i = 0; i < info_gs.it; ++i) {
        std::cout << info_gs.history[i] << '\n';
    }

    return 0;
}