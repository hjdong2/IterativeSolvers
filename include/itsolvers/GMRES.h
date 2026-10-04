
#pragma once
#include <Eigen/Dense>
#include "SolverTypes.h"
#include "Arnoldi.h"

namespace itsolvers {

inline SolverResult gmres(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd & x, const SolverOptions& opts) {
    Eigen::VectorXd r0 = b - A * x;
    std::vector<double> history;

    double beta = r0.norm();
    if (beta <= 1e-12 * b.norm()) {
        SolverResult info = {0.0, 0, true, history};
        return info; 
    }

    Eigen::Index m = opts.m;

    auto arnores = itsolvers::arnoldi(A, r0, m);
    Eigen::MatrixXd Hmbar = arnores.Hmbar;
    Eigen::MatrixXd Vmp1 = arnores.Vmp1;
    Eigen::Index k = arnores.k;

    Eigen::VectorXd e1 = Eigen::VectorXd::Unit(k + 1, 0);

    Eigen::VectorXd y = Hmbar.topLeftCorner(k + 1, k).householderQr().solve(beta * e1);

    Eigen::MatrixXd Vk = Vmp1.leftCols(k);
    x += Vk * y;

    double res = (b - A * x).norm() / b.norm();
    bool isconverged = (res <= opts.reltol) ? true : false;
    int it = k; 
    SolverResult info = {res, it, isconverged, history};

    return info;
}

}