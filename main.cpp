
#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "Solver.h"

Eigen::MatrixXd classicGramSchmist(const Eigen::MatrixXd& A) {
    int m = A.rows(), n = A.cols();
    Eigen::MatrixXd Q(m, n);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n, n);

    for (int j = 0; j < n; ++j) {
        Eigen::VectorXd q = A.col(j);
        Eigen::VectorXd temp = Eigen::VectorXd::Zero(m);
        for (int i = 0; i < j; ++i) {
            double rij = q.dot(Q.col(i));
            temp += rij * Q.col(i);
        }
        q = q - temp;
        R(j, j) = q.norm();
        q /= R(j, j);
        Q.col(j) = q;        
    }

    return Q;
}

Eigen::MatrixXd modifiedGramSchmidt(const Eigen::MatrixXd& A) {
    int m = A.rows(), n = A.cols();
    Eigen::MatrixXd Q(m, n);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n, n);

    for (int j = 0; j < n; ++j) {
        Eigen::VectorXd q = A.col(j);
        for (int i = 0; i < j; ++i) {
            double rij = q.dot(Q.col(i));
            q = q - rij * Q.col(i);
        }
        R(j, j) = q.norm();
        q /= R(j, j);
        Q.col(j) = q;
    }

    return Q;
}


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

    Eigen::MatrixXd A_eig(2, 2);
    A_eig << 1, 2, 3, 4;
    std::cout << A_eig << '\n';

    return 0;
}