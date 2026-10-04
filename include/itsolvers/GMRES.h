
#pragma once
#include <Eigen/Dense>
#include <cmath>
#include "SolverTypes.h"
#include "Arnoldi.h"

namespace itsolvers {

inline SolverResult gmres(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd & x, const SolverOptions& opts) {
    Eigen::VectorXd r0 = b - A * x;
    std::vector<double> history;

    if (std::abs(b.norm()) < 1e-16) {
        x.setZero();
        SolverResult info = {0.0, 0, true, history};
        return info;
    }

    double beta = r0.norm();
    if (beta <= opts.reltol * b.norm()) {
        SolverResult info = {0.0, 0, true, history};
        return info; 
    }

    Eigen::Index m = opts.m;
    Eigen::Index n = A.rows();
    Eigen::MatrixXd Vmp1 = Eigen::MatrixXd::Zero(n, m + 1);
    Eigen::MatrixXd Hmbar = Eigen::MatrixXd::Zero(m + 1, m);
    Eigen::MatrixXd GivRot = Eigen::MatrixXd::Zero(2, m);
    Eigen::VectorXd e1 = Eigen::VectorXd::Unit(m + 1, 0);
    Eigen::VectorXd g = beta * e1;
    Vmp1.col(0) = r0 / beta;

    Eigen::Index k = 0;

    for (Eigen::Index j = 0; j < m; ++j) {
        bool hasNewVec = arnoldiStep(A, Vmp1, Hmbar, j);
        for (Eigen::Index i = 0; i < j; ++i) {
            double temp = GivRot(0, i) * Hmbar(i, j) + GivRot(1, i) * Hmbar(i + 1, j);
            Hmbar(i + 1, j) = -GivRot(1, i) * Hmbar(i, j) + GivRot(0, i) * Hmbar(i + 1, j);
            Hmbar(i, j) = temp;
        }
        double r = std::hypot(Hmbar(j, j), Hmbar(j + 1, j));
        double c = Hmbar(j, j) / r;
        double s = Hmbar(j + 1, j) / r;
        GivRot.col(j) << c, s;

        double temp = c * Hmbar(j, j) + s * Hmbar(j + 1, j);
        Hmbar(j + 1, j) = -s * Hmbar(j, j) + c * Hmbar(j + 1, j);
        Hmbar(j, j) = temp;
        g(j + 1) = -s * g(j);
        g(j) = c * g(j);

        history.push_back(std::abs(g(j + 1)) / b.norm());
        k = j + 1;
        if (std::abs(g(j + 1)) <= opts.reltol * b.norm() || !hasNewVec) break;
    }

    Eigen::VectorXd y = Hmbar.topLeftCorner(k, k).triangularView<Eigen::Upper>().solve(g.head(k));
    x += Vmp1.leftCols(k) * y;

    double res = std::abs(g(k)) / b.norm();
    SolverResult info = {res, static_cast<int>(k), res <= opts.reltol, history};

    return info;

}

}