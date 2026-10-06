
#include <iostream>
#include <itsolvers/GMRES.h>

int main() {

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(10, 10);
    Eigen::VectorXd main_diag = Eigen::VectorXd::Constant(10, 4.0);
    Eigen::VectorXd sub_diag = Eigen::VectorXd::Constant(10 - 1, -1.0);

    A.diagonal(0) = main_diag;
    A.diagonal(1) = sub_diag;
    A.diagonal(-1) = sub_diag;

    Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(10, 1.0, 10.0);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(10);

    itsolvers::SolverOptions opts = {1e-14, 1000, 1.0, 10};
    auto info = itsolvers::gmres(A, b, x, opts);
    for (int i = 0; i < info.it; ++i) {
        std::cout << "Step i= " << i + 1 << ",     Givens residual= " << info.history[i] << '\n';
    }

    std::vector<int> nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::cout << "============= Givens residual vs.true residual for tridiagonal 10x10 ===============\n";
    for (auto k : nums) {
        x = Eigen::VectorXd::Zero(10);
        itsolvers::SolverOptions optk = {1e-14, 1000, 1.0, k};
        auto infok = itsolvers::gmres(A, b, x, optk);

        std::cout << "Step k= " << k << ",    Iterations= " << infok.it << 
                ",    Givens residual= " << infok.res << ",    True residual= " << (b - A * x).norm() / b.norm() << '\n';
    }
    return 0;
}
