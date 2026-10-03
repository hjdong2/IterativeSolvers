
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
    checkNear((x - xext).norm(), 0.0, 1e-14, "GMRES 3x3: solution matchs well direct solve");

}

int main() {
    testGmresTridiagMatrix();

    std::cout << (checks - failures) << "/" << checks << " checks passed\n";

    return failures == 0 ? 0 : 1;
}