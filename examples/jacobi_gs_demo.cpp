
#include <iostream>
#include <vector>
#include <itsolvers/Solver.h>

int main() {

    itsolvers::SolverOptions opts = {1e-8, 1000, 1.0};
    int n = 10;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd main_diag = Eigen::VectorXd::Constant(n, 4.0);
    Eigen::VectorXd sub_diag = Eigen::VectorXd::Constant(n - 1, -1.0);

    A.diagonal(0) = main_diag;
    A.diagonal(1) = sub_diag;
    A.diagonal(-1) = sub_diag;

    Eigen::VectorXd x_ja = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd x_gs = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd b(n);
    b << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0;

    Eigen::VectorXd xext = A.partialPivLu().solve(b);

    itsolvers::SolverResult info_ja = itsolvers::jacobi(A, b, x_ja, opts);
    itsolvers::SolverResult info_gs = itsolvers::GaussSeidel(A, b, x_gs, opts);

    std::cout << "================ Test for Jacobi ================\n";
    std::cout << "Isconverged=        " << info_ja.isconverged << '\n';
    std::cout << "Iteration num=      " << info_ja.it << '\n';
    std::cout << "Residual=           " << info_ja.res << '\n';
    std::cout << "nrm(x-x_ext)=       " << (x_ja - xext).norm() << '\n'; 


    std::cout << "================ Test for Gauss Seidel ================\n";
    std::cout << "Isconverged=     " << info_gs.isconverged << '\n';
    std::cout << "Iteration num=   " << info_gs.it << '\n';
    std::cout << "Residual=        " << info_gs.res << '\n';
    std::cout << "nrm(x-x_ext)=    " << (x_gs - xext).norm() << '\n'; 

    return 0;
}