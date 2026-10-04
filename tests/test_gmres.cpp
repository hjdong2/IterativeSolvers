
#include <iostream>
#include <itsolvers/GMRES.h>
#include "test_helpers.h"

void testGmresTridiagMatrix() {
    Eigen::MatrixXd A = makeTridiag(10);
    Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(10, 1.0, 10.0);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(10);

    itsolvers::SolverOptions opts = {1e-8, 10000, 1.0, 10};
    itsolvers::SolverResult info = itsolvers::gmres(A, b, x, opts);

    Eigen::VectorXd xext = A.partialPivLu().solve(b);
    checkNear((x - xext).norm(), 0.0, 1e-14, "GMRES tridiag: soultion match well with exact solution");  

}

void testGmres3x3Matrix() {
    Eigen::MatrixXd A = make3x3();
    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(3);

    itsolvers::SolverOptions opts = {1e-8, 10000, 1.0, 3};
    itsolvers::SolverResult info = itsolvers::gmres(A, b, x, opts);

    Eigen::VectorXd xext = A.partialPivLu().solve(b);
    checkNear((x - xext).norm(), 0.0, 1e-10, "GMRES 3x3: solution matchs well direct solve");

}

void testGmresIdentityMatrix() {
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(5, 5);
    Eigen::VectorXd b = Eigen::VectorXd::Ones(5);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(5);

    itsolvers::SolverOptions opts = {1e-8, 100, 1.0, 3};
    itsolvers::SolverResult info = itsolvers::gmres(A, b, x, opts);

    Eigen::VectorXd xext = A.partialPivLu().solve(b);

    checkNear((x - xext).norm(), 0.0, 1e-10, "GMRES Breakdown");

}

void testGmresBreakdown() {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(10, 10);
    A.diagonal() << 1, 2, 3, 1, 2, 3, 1, 2, 3, 1;

    Eigen::VectorXd b = Eigen::VectorXd::Ones(10);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(10);

    itsolvers::SolverOptions opts = {1e-8, 100, 1.0, 6};
    itsolvers::SolverResult info = itsolvers::gmres(A, b, x, opts);

    Eigen::VectorXd xext = A.partialPivLu().solve(b);

    checkNear((x - xext).norm(), 0.0, 1e-10, "GMRES Breakdown");
    check(info.it == 3, "GMRES breakdown");

}

void testGmresSubspace() {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(10, 10);
    Eigen::VectorXd main_diag = Eigen::VectorXd::Constant(10, 1.0);
    Eigen::VectorXd upp_diag = Eigen::VectorXd::Constant(10 - 1, 2.0);
    Eigen::VectorXd low_diag = Eigen::VectorXd::Constant(10 - 1, 3.0);

    A.diagonal(0) = main_diag;
    A.diagonal(1) = upp_diag;
    A.diagonal(-1) = low_diag;

    Eigen::VectorXd b = Eigen::VectorXd::Ones(10);

    Eigen::VectorXd x2 = Eigen::VectorXd::Zero(10);
    Eigen::VectorXd x4 = Eigen::VectorXd::Zero(10);
    Eigen::VectorXd x6 = Eigen::VectorXd::Zero(10);
    Eigen::VectorXd x8 = Eigen::VectorXd::Zero(10);
    Eigen::VectorXd x10 = Eigen::VectorXd::Zero(10);

    itsolvers::SolverOptions opts2 = {1e-8, 100, 1.0, 2};
    itsolvers::SolverOptions opts4 = {1e-8, 100, 1.0, 4};
    itsolvers::SolverOptions opts6 = {1e-8, 100, 1.0, 6};
    itsolvers::SolverOptions opts8 = {1e-8, 100, 1.0, 8};
    itsolvers::SolverOptions opts10 = {1e-8, 100, 1.0, 10};

    itsolvers::SolverResult info2 = itsolvers::gmres(A, b, x2, opts2);
    itsolvers::SolverResult info4 = itsolvers::gmres(A, b, x4, opts4);
    itsolvers::SolverResult info6 = itsolvers::gmres(A, b, x6, opts6);
    itsolvers::SolverResult info8 = itsolvers::gmres(A, b, x8, opts8);
    itsolvers::SolverResult info10 = itsolvers::gmres(A, b, x10, opts10);

    Eigen::VectorXd xext = A.partialPivLu().solve(b);

    checkNear((x2 - xext).norm(), 0.0, 1e-10, "GMRES m=2: Breakdown");
    checkNear((x4 - xext).norm(), 0.0, 1e-10, "GMRES m=4: Breakdown");
    checkNear((x6 - xext).norm(), 0.0, 1e-10, "GMRES m=6: Breakdown");
    checkNear((x8 - xext).norm(), 0.0, 1e-10, "GMRES m=8: Breakdown");
    checkNear((x10 - xext).norm(), 0.0, 1e-10, "GMRES m=10: Breakdown");

}

int main() {
    testGmresTridiagMatrix();
    testGmres3x3Matrix();
    testGmresIdentityMatrix();
    testGmresBreakdown();
    testGmresSubspace();

    std::cout << (checks - failures) << "/" << checks << " checks passed\n";

    return failures == 0 ? 0 : 1;
}