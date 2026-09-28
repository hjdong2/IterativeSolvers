
#pragma once
#include <Eigen/Dense>
#include <tuple>

namespace itsolvers {
inline std::tuple<Eigen::MatrixXd, Eigen::MatrixXd > classicGramSchmidt(const Eigen::MatrixXd& A) {
    Eigen::Index m = A.rows(), n = A.cols();
    Eigen::MatrixXd Q(m, n);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n, n);

    for (Eigen::Index j = 0; j < n; ++j) {
        Eigen::VectorXd q = A.col(j);
        Eigen::VectorXd temp = Eigen::VectorXd::Zero(m);
        for (Eigen::Index i = 0; i < j; ++i) {
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

inline std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> modifiedGramSchmidt(const Eigen::MatrixXd& A) {
    Eigen::Index m = A.rows(), n = A.cols();
    Eigen::MatrixXd Q(m, n);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n, n);

    for (Eigen::Index j = 0; j < n; ++j) {
        Eigen::VectorXd q = A.col(j);
        for (Eigen::Index i = 0; i < j; ++i) {
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

}