
#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <tuple>
#include "Solver.h"

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd > classicGramSchmidt(const Eigen::MatrixXd& A) {
    int m = A.rows(), n = A.cols();
    Eigen::MatrixXd Q(m, n);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n, n);

    for (int j = 0; j < n; ++j) {
        Eigen::VectorXd q = A.col(j);
        Eigen::VectorXd temp = Eigen::VectorXd::Zero(m);
        for (int i = 0; i < j; ++i) {
            double rij = q.dot(Q.col(i));
            R(i, j) = rij;
            temp += rij * Q.col(i);
        }
        q = q - temp;
        R(j, j) = q.norm();
        q /= R(j, j);
        Q.col(j) = q;        
    }

    return {Q, R};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> modifiedGramSchmidt(const Eigen::MatrixXd& A) {
    int m = A.rows(), n = A.cols();
    Eigen::MatrixXd Q(m, n);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n, n);

    for (int j = 0; j < n; ++j) {
        Eigen::VectorXd q = A.col(j);
        for (int i = 0; i < j; ++i) {
            double rij = q.dot(Q.col(i));
            R(i, j) = rij;
            q = q - rij * Q.col(i);
        }
        R(j, j) = q.norm();
        q /= R(j, j);
        Q.col(j) = q;
    }

    return {Q, R};
}

// TODO:
// 1. still need to think about the structure
// 2. Linear solver information
// 3. Ask 

int main() {
    std::vector<std::vector<double>> A = {{4.0, 1.0}, {1.0, 3.0}};
    std::vector<double> b = {1.0, 2.0};
    std::vector<double> x = {0.0, 0.0};

    // SolverOptions opts = {1e-16, 1000, 0.0};
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

    // SolverResult info_gs = GaussSeidel(A, b, x, opts);
    // std::cout << info_gs.isconverged << '\n';
    // std::cout << "Res= " << info_gs.res << '\n';
    // std::cout << "it= " << info_gs.it << '\n';
    // for (int i = 0; i < A.size(); ++i) {
    //     std::cout << x[i] << '\n';
    // }

    // for (int i = 0; i < info_gs.it; ++i) {
    //     std::cout << info_gs.history[i] << '\n';
    // }
    
    std::vector<int> nums = {4, 6, 8, 10, 12};
    std::cout << "n" << "     " << "   CGS   " << "     " << "   MGS   " << '\n';
    for (auto n : nums) {
        Eigen::MatrixXd H(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                H(i, j) = 1.0 / (i + j + 1.0);
            }
        }
        // std::cout << H << '\n';
        auto [Qc, Rc] = classicGramSchmidt(H);
        auto [Qm, Rm] = modifiedGramSchmidt(H);
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
        double nrmc = (I - Qc.transpose() * Qc).norm();
        double nrmm = (I - Qm.transpose() * Qm).norm();
        
        std::cout << n << "    " << nrmc << "    " << nrmm << '\n'; 
    }
    

    return 0;
}