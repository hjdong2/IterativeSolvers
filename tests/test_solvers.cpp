
#include <iostream>
#include <itsolvers/Solver.h>
#include "test_helpers.h"

void testGaussSeidelConvergesOnTridiag() {
    Eigen::MatrixXd A = makeTridiag(10);
    Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(10, 1.0, 10.0);
    Eigen::VectorXd x_gs = Eigen::VectorXd::Zero(10);

    itsolvers::SolverOptions opts;
    itsolvers::SolverResult info_gs = itsolvers::GaussSeidel(A, b, x_gs, opts);

    Eigen::VectorXd xext = A.partialPivLu().solve(b);

    check(info_gs.isconverged, "GS tridiag: converged");
    check(info_gs.res <= opts.reltol, "GS tridiag: final res below tol");
    checkNear((x_gs - xext).norm(), 0.0, 1e-6, "GS tridiag: solution match well direct solve");

}

void testGaussSeidelFasterThanJacobi() {
    Eigen::MatrixXd A = makeTridiag(10);
    Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(10, 1.0, 10.0);
    Eigen::VectorXd x_gs = Eigen::VectorXd::Zero(10);
    Eigen::VectorXd x_ja = Eigen::VectorXd::Zero(10);

    itsolvers::SolverOptions opts;
    itsolvers::SolverResult info_gs = itsolvers::GaussSeidel(A, b, x_gs, opts);
    itsolvers::SolverResult info_ja = itsolvers::jacobi(A, b, x_ja, opts);

    check(info_gs.isconverged, "GS tridiag: converged");
    check(info_ja.isconverged, "Jacobi tridiag: converged");
    check((info_gs.it < info_ja.it), "GS needs fewer iterations than Jacobi"); 

}

void testJacobiDiverge3x3Matrix() {
    Eigen::MatrixXd A = make3x3();
    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(3);

    itsolvers::SolverOptions opts= {1e-8, 50, 1.0};
    itsolvers::SolverResult info = itsolvers::jacobi(A, b, x, opts);

    check(!info.isconverged, "Jacobi 3x3: diverges (nor converged)");
    check(info.res > 1.0, "Jacobi 3x3: residual grew");
    check(info.it == opts.maxit, "Jacobi maximum iteration");
}

void testGSConverge3x3Matrix() {
    Eigen::MatrixXd A = make3x3();
    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(3);

    itsolvers::SolverOptions opts;
    itsolvers::SolverResult info = itsolvers::GaussSeidel(A, b, x, opts);

    Eigen::VectorXd xext = A.partialPivLu().solve(b);

    check(info.isconverged, "GS 3x3: converged");
    checkNear((x - xext).norm(), 0.0, 1e-6, "GS 3x3: solution matchs well direct solve");
}

void testGSbIsZeros() {
    Eigen::MatrixXd A = make3x3();
    Eigen::VectorXd b = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x = Eigen::VectorXd::Ones(3);

    itsolvers::SolverOptions opts;
    itsolvers::SolverResult info = itsolvers::GaussSeidel(A, b, x, opts);

    check(info.isconverged, "GS for b is zero: converged");
    check(x.norm() == 0.0, "GS b =0: x set to zero");
    check(info.res == 0.0, "GS residual is zero");
}

void testjacobibIsZeros() {
    Eigen::MatrixXd A = make3x3();
    Eigen::VectorXd b = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x = Eigen::VectorXd::Ones(3);

    itsolvers::SolverOptions opts;
    itsolvers::SolverResult info = itsolvers::jacobi(A, b, x, opts);

    check(info.isconverged, "jacobi for b is zero: converged");
    check(x.norm() == 0.0, "jacobi b = 0: x set to zero");
    check(info.res == 0.0, "jacobi residual is zero");
}

int main() {
    testGaussSeidelConvergesOnTridiag();
    testGaussSeidelFasterThanJacobi();
    testJacobiDiverge3x3Matrix();
    testGSConverge3x3Matrix();
    testGSbIsZeros();
    testjacobibIsZeros();

    std::cout << (checks - failures) << "/" << checks << " checks passed\n";

    return failures == 0 ? 0 : 1;
}