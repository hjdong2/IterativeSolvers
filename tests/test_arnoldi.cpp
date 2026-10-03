
#include <iostream>
#include <itsolvers/Arnoldi.h>
#include "test_helpers.h"

// ================= test Arnoldi algorithm ================
void testMatrixOrthonormality() {
    Eigen::MatrixXd A = makeTridiag(10);
    Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(10, 1.0, 10.0);

    Eigen::Index m = 5;
    Eigen::VectorXd r0 = b;

    auto res = itsolvers::arnoldi(A, r0, m);
    Eigen::MatrixXd Vmp1 = res.Vmp1;
    Eigen::MatrixXd Hmbar = res.Hmbar;
    Eigen::Index nvalid = (res.k < m) ? res.k : res.k + 1;

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nvalid, nvalid);
    Eigen::MatrixXd Q = Vmp1.leftCols(nvalid);

    checkNear((I - Q.transpose() * Q).norm(), 0.0, 1e-12, "Arnoldi tridiag: orthonormal basis");
    check(res.k == m, "Arnoldi tridiag: no breakdown");
}

void testArnoldiRelation() {
    Eigen::MatrixXd A = makeTridiag(10);
    Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(10, 1.0, 10.0);

    Eigen::Index m = 5;
    Eigen::VectorXd r0 = b;

    auto res = itsolvers::arnoldi(A, r0, m);
    Eigen::MatrixXd Vmp1 = res.Vmp1;
    Eigen::MatrixXd Hmbar = res.Hmbar;
    Eigen::Index nvalid = (res.k < m) ? res.k : res.k + 1;

    Eigen::MatrixXd Vk = Vmp1.leftCols(nvalid - 1);
    Eigen::MatrixXd Vkp1 = Vmp1.leftCols(nvalid);

    checkNear((A * Vk - Vkp1 * Hmbar).norm() / A.norm(), 0.0, 1e-12, "Arnoldi Relation matchs");

}

void testDoubleCheckNonSymmetricMatrix() {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(10, 10);
    Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(10, 1.0, 10.0);

    Eigen::Index m = 5;
    Eigen::VectorXd r0 = b;

    auto res = itsolvers::arnoldi(A, r0, m);
    Eigen::MatrixXd Vmp1 = res.Vmp1;
    Eigen::MatrixXd Hmbar = res.Hmbar;
    Eigen::Index nvalid = (res.k < m) ? res.k : res.k + 1;

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nvalid, nvalid);
    Eigen::MatrixXd Q = Vmp1.leftCols(nvalid);

    checkNear((I - Q.transpose() * Q).norm(), 0.0, 1e-12, "Arnoldi tridiag: orthonormal basis");
    check(res.k == m, "Arnoldi tridiag: no breakdown");

    Eigen::MatrixXd Vk = Vmp1.leftCols(nvalid - 1);
    Eigen::MatrixXd Vkp1 = Vmp1.leftCols(nvalid);

    checkNear((A * Vk - Vkp1 * Hmbar).norm() / A.norm(), 0.0, 1e-12, "Arnoldi Relation matchs");

}

void testIdentityMatrix() {
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(10, 10);
    Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(10, 1.0, 10.0);

    Eigen::Index m = 5;
    Eigen::VectorXd r0 = b;

    auto res = itsolvers::arnoldi(A, r0, m);
    Eigen::MatrixXd Vmp1 = res.Vmp1;
    Eigen::MatrixXd Hmbar = res.Hmbar;
    Eigen::Index nvalid = (res.k < m) ? res.k : res.k + 1;

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nvalid, nvalid);
    Eigen::MatrixXd Q = Vmp1.leftCols(nvalid);

    std::cout << "res.k= " << res.k << '\n';
    std::cout << "m= " << m << '\n';
    checkNear((I - Q.transpose() * Q).norm(), 0.0, 1e-12, "Arnoldi tridiag: orthonormal basis");
    check(res.k == 1, "Arnoldi tridiag: no breakdown");

}

int main() {
    testMatrixOrthonormality();
    testArnoldiRelation();
    testDoubleCheckNonSymmetricMatrix();
    testIdentityMatrix();

    std::cout << (checks - failures) << "/" << checks << " checks passed\n";

    return failures == 0 ? 0 : 1;
    
}