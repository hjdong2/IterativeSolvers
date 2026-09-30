
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

    Eigen::VectorXd v0 = r0 / r0.norm();
    Vmp1.col(0) = v0;
    for (Eigen::Index j = 0; j < m; ++j) {
        Eigen::VectorXd wj = A * Vmp1.col(j);
        for (Eigen::Index i = 0; i <= j; ++i){
            double hij = wj.dot(Vmp1.col(i));
            wj -= hij * Vmp1.col(i);
            Hmbar(i, j) = hij;
        }
        hjp1j = wj.norm();
        if (std::abs(hjp1j) > 1e-8) {
            Hmbar(j + 1, j) = hjp1j;
            Vmp1.col(j) = wj / hjp1j;
        } else {
            return Arnoldi {Vmp1, Hmbar, j};
        }
    }

    return Arnoldi {Vmp1, Hmbar, m};

}


}