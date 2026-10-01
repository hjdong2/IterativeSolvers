
#include <iostream>
#include <itsolvers/Arnoldi.h>
#include <tests/test_helpers.h>

int failures = 0;
int checks = 0;

// ================= test Arnoldi algorithm ================
void testMatrixOrthonormality() {
    Eigen::MatrixXd A = makeTridiag(10);
    Eigen::VectorXd b = Eigen::VectorXd::Linspace(10, 1.0, 10.0);

    Eigen::Index m = 5;
    Eigen::VectorXd r0 = b;

    auto res = arnoldi(A, r0, b);
    Eigen::MatrixXd Vmp1 = res.Vmp1;
    Eigen::MatrixXd Hmbar = res.Hmbar;
    Eigen::Index nvalid = (res.k < m) ? res.k : res.k + 1;

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nvalid, nvalid);
    Eigen::MatrixXd Q = Vmp1.leftCols(nvalid);

    checkNear((I - Q.transpose() * Q).norm(), 0.0, 1e-14, "Arnoldi Orthonormality")
    chekc(nvalid == m, "Arnoldi breakdown");
}

void testArnoldiRelation() {
    Eigen::MatrixXd A = makeTridiag(10);
    Eigen::VectorXd b = Eigen::VectorXd::Linspace(10, 1.0, 10.0);

    Eigen::Index m = 5;
    Eigen::VectorXd r0 = b;

    auto res = arnoldi(A, r0, b);
    Eigen::MatrixXd Vmp1 = res.Vmp1;
    Eigen::MatrixXd Hmbar = res.Hmbar;
    Eigen::Index nvalid = (res.k < m) ? res.k : res.k + 1;

    Eigen::MatrixXd Vk = Vmp1.Cols(nvalid);
    Eigen::MatrixXd Vkp1 = Vmp1.Cols(nvalid + 1);

    checkNear((A * Vk - Vkp1 * Hmbar).norm() / A.norm(), 0.0, 1e-14, "Arnoldi Relation matchs")

}

void testDoubleCheckNonSymmetricMatrix() {
    Eigen::MatrixXd A = Eigen::Random(10, 10);

}

void testHessenbergUpperMatrix() {
    Eigen::MatrixXd A = makeTridiag(10);
    Eigen::VectorXd b = Eigen::VectorXd::Linspace(10, 1.0, 10.0);

    Eigen::Index m = 5;
    Eigen::VectorXd r0 = b;

    auto res = arnoldi(A, r0, b);
    Eigen::MatrixXd Vmp1 = res.Vmp1;
    Eigen::MatrixXd Hmbar = res.Hmbar;

    Eigen::Index row = Hmbar.rows();
    Eigen::Index col = Hmbar.cols();

    for (Eigen::Index j = 0; j < col; ++j) {
        for (Eigen::Index i = j + 1; i < row; ++i) {
            if (H(i, j) == 0.0) {
                continue;
            } else {
                std::cout << "Hessenbers is not uppder triangular\n";
            }
        }
    }

}

void testArnoldiBreakdown() {
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(5, 5);
    Eigen::VectorXd r0 = Eigen::VectorXd::Ones(5);
    Eigen::Index m = 3;

    auto res = arnoldi(A, r0, m);

    
}

void testBreakdownAfter3Steps() {

}

void testr0IsZero() {

}

void testFirstBasisNormalize() {

}

void testScaling() {
    
}