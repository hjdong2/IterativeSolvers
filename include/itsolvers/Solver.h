
#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>
#include <utility>
#include <Eigen/Dense>

namespace itsolvers {

struct SolverOptions{
    double reltol=1e-8;
    int maxit=10000;
    double w=1.0;     // for SOR relaxation
};

struct SolverResult{
    double res;
    int it;
    bool isconverged;
    std::vector<double> history;
};

inline SolverResult jacobi(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd& x, const SolverOptions& opts) {
    Eigen::Index n = A.rows();

    double reltol = opts.reltol;
    int maxit = opts.maxit;

    double res = {0.0};
    int nit = maxit;
    bool isconverged = false;
    std::vector<double> history;

    SolverResult info = {res, nit, isconverged, history};

    Eigen::VectorXd x_new = Eigen::VectorXd::Zero(n);
    double bnrm = b.norm();
    if (std::abs(bnrm) < 1e-16) {
        x.setZero();
        info.isconverged = true;
        info.it = 0;
        info.history.push_back(0.0);
        return info;
    }

    for (Eigen::Index it = 0; it < maxit; ++it) {
        for (Eigen::Index i = 0; i < n; ++i) {
            double num = 0.0;
            for (Eigen::Index j = 0; j < n; ++j) {
                if (i != j) num += A(i, j) * x(j);
            }
            if (std::abs(A(i, i)) <= 1e-16) {
                info.isconverged = false;
                return info;
            }
            x_new(i) = (-num + b(i)) / A(i, i);
        }
        swap(x, x_new);

        double r = (b - A * x).norm();
        info.history.push_back(r / bnrm);

        if (r / bnrm <= reltol) {
            info.isconverged = true;
            info.it = it + 1;
            info.res = r / bnrm;
            return info;
        }
    }

    info.isconverged = false;
    info.it = maxit;
    info.res = info.history.back();

    return info;

}

inline SolverResult GaussSeidel(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd& x, const SolverOptions& opts) {
    double reltol = opts.reltol;
    int maxit = opts.maxit;

    Eigen::Index n = A.rows();
    std::vector<double> history;
    SolverResult info = {0.0, maxit, false, history};

    Eigen::VectorXd x_new = Eigen::VectorXd::Zero(n);
    double bnrm = b.norm();
    if (std::abs(bnrm) < 1e-16) {
        x.setZero();
        info.isconverged = true;
        info.it = 0;
        info.history.push_back(0.0);
        return info;
    }

    for (Eigen::Index it = 0; it < maxit; ++it) {
        for (Eigen::Index i = 0; i < n; ++i) {
            double num = 0.0;
            for (Eigen::Index j = 0; j <= i - 1; ++j) {
                num += A(i, j) * x_new(j);
            }

            for (Eigen::Index j = i + 1; j < n; ++j) {
                num += A(i, j) * x(j);
            }

            if (std::abs(A(i, i)) < 1e-16) {
                info.isconverged = false;
                return info;
            }

            x_new(i) = (b(i) - num) / A(i, i);
        }
        swap(x, x_new);

        double r = (b - A * x).norm();
        info.history.push_back(r / bnrm);

        if ((r / bnrm) <= reltol) {
            info.isconverged = true;
            info.it = it + 1;
            info.res = r / bnrm;
            return info;
        }
    }

    info.isconverged = false;
    info.it = maxit;
    info.res = info.history.back();

    return info;
}

}