
#pragma once
#include <Eigen/Dense>

namespace itsolvers {

struct ArnoldiResult {
    Eigen::MatrixXd Vmp1;
    Eigen::MatrixXd Hmbar;
    Eigen::Index k;
};

inline ArnoldiResult arnoldi (const Eigen::MatrixXd& A, const Eigen::VectorXd& r0, Eigen::Index m) {
    Eigen::Index n = A.rows();
    Eigen::MatrixXd Vmp1 = Eigen::MatrixXd Zero(n, m + 1);
    Eigen::MatrixXd Hmbar = Eigen::MatrixXd Zero(m + 1, m);

    double tol = 1e-12;

    if (r0.norm() == 0.0) {
        ArnoldiResult res = {Vmp1, Hmbar, 0};
        return res;
    }
    Eigen::VectorXd v0 = r0 / r0.norm();
    Vmp1.col(0) = v0;
    for (Eigen::Index j = 0; j < m; ++j) {
        Eigen::VectorXd wj = A * Vmp1.col(j);
        double wnrm = wj.norm();
        for (Eigen::Index i = 0; i <= j; ++i){
            double hij = wj.dot(Vmp1.col(i));
            wj -= hij * Vmp1.col(i);
            Hmbar(i, j) = hij;
        }
        hjp1j = wj.norm();
        if (hjp1j > tol * wnrm) {
            Hmbar(j + 1, j) = hjp1j;
            Vmp1.col(j + 1) = wj / hjp1j;
        } else {
            Eigen::Index k = j;
            ArnoldiResult res = {Vmp1, Hmbar, k};
            return res;
        }
    }

    ArnoldiResult res = {Vmp1, Hmbar, m};

    return res;

}


}