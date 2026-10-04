
#pragma once
#include <Eigen/Dense>

namespace itsolvers {

struct ArnoldiResult {
    Eigen::MatrixXd Vmp1;
    Eigen::MatrixXd Hmbar;
    Eigen::Index k;
};

inline bool arnoldiStep(const Eigen::MatrixXd& A, Eigen::MatrixXd& V, Eigen::MatrixXd& H, Eigen::Index j) {
    Eigen::VectorXd wj = A * V.col(j);
    double wnrm = wj.norm();
    double tol = 1e-12;

    for (Eigen::Index i = 0; i <= j; ++i) {
        double hij = wj.dot(V.col(i));
        wj -= hij * V.col(i);
        H(i, j) = hij;
    }

    double hjp1j = wj.norm();
    if (hjp1j > tol * wnrm) {
        H(j + 1, j) = hjp1j;
        V.col(j + 1) = wj / hjp1j;
        return true;
    }

    return false;
}

inline ArnoldiResult arnoldi (const Eigen::MatrixXd& A, const Eigen::VectorXd& r0, Eigen::Index m) {
    Eigen::Index n = A.rows();
    Eigen::MatrixXd Vmp1 = Eigen::MatrixXd::Zero(n, m + 1);
    Eigen::MatrixXd Hmbar = Eigen::MatrixXd::Zero(m + 1, m);

    if (r0.norm() == 0.0) {
        ArnoldiResult res = {Vmp1, Hmbar, 0};
        return res;
    }
    Eigen::VectorXd v0 = r0 / r0.norm();
    Vmp1.col(0) = v0;
    for (Eigen::Index j = 0; j < m; ++j) {
        bool hasNewVec = arnoldiStep(A, Vmp1, Hmbar, j);
        if (!hasNewVec) {
            return {Vmp1, Hmbar, j + 1};
        }
    }

    ArnoldiResult res = {Vmp1, Hmbar, m};

    return res;

}


}