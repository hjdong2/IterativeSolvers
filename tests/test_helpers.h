
#pragma once
#include <iostream>
#include <cmath>

inline int failures = 0;
inline int checks = 0;

inline void check(bool ok, const char* name) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cout << "FAILED: " << name << '\n';
    }
}

inline void checkNear(double actual, double expected, double tol, const char* name) {
    ++checks;
    if (std::abs(actual - expected) > tol) {
        ++failures;
        std::cout << "FAILED: " << name << 
                    "(got " << actual << ", expected " <<
                    expected << ")\n";
    }
}

// ================= test matrices ===================
inline Eigen::MatrixXd makeTridiag(int n) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd main_diag = Eigen::VectorXd::Constant(n, 4.0);
    Eigen::VectorXd sub_diag = Eigen::VectorXd::Constant(n - 1, -1.0);

    A.diagonal(0) = main_diag;
    A.diagonal(1) = sub_diag;
    A.diagonal(-1) = sub_diag;

    return A;    
}

inline Eigen::MatrixXd make3x3() {
    Eigen::MatrixXd A(3, 3);
    A << 1.0, 3.0, 5.0,
         4.0, 9.5, 8.0,
         1.0, 4.0, 4.0;

    return A;
}
