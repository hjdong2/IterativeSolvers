
#pragma once
#include <Eigen/Dense>
#include "SolverTypes.h"
#include "Arnoldi.h"

namespace itsolvers {

inline SolverResult gmres(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd & x, const SolverOptions& opts) {
    Eigen::VectorXd r0 = b - A * x;
    double beta = r0.norm();
    Eigen::VectorXd v0 = r0 / beta;
    Eigen::Index m = opts.m;

    auto arnores = itsolvers::arnoldi(A, r0, m);
    Eigen::MatrixXd Hmbar = arnores.Hmbar;
    Eigen::MatrixXd Vmp1 = arnores.Vmp1;
    Eigen::Index nvalid = (arnores.k < m) ? arnores.k : arnores.k + 1;

    Eigen::VectorXd e1 = Eigen::VectorXd::Unit(nvalid, 0.0);

    Eigen::VectorXd y = Hmbar.topLeftCorner(nvalid, nvalid - 1).householderQr().solve(beta * e1);

    Eigen::MatrixXd Vk = Vmp1.leftCols(nvalid - 1);
    x += Vk * y;

    int maxit = opts.maxit;
    std::vector<double> history;
    SolverResult info = {0.0, maxit, false, history};

    return info;
}

}