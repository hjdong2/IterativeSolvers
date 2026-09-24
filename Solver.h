
#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>

struct SolverOptions{
    double reltol=1e-8;
    int maxit=10000;
    double w=1.0;     // for SOR relaxation
};

struct SolverResult{
    double res;
    int it;
    bool isconverged;
    std::vector<double> history;
};

inline double resNrm(const std::vector<std::vector<double>>& A, const std::vector<double>& b, const std::vector<double>& x)  {
    int n = A.size();
    std::vector<double> r(n);
    for (int i = 0; i < n; ++i) {
        double num = 0.0;
        for (int j = 0; j < n; ++j) {
            num += A[i][j] * x[j];
        }
        r[i] = b[i] - num;
    }

    return std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0.0));

}

inline SolverResult jacobi(const std::vector<std::vector<double>>& A, const std::vector<double>& b,
        std::vector<double>& x, const SolverOptions& opts) {
            double reltol = opts.reltol;
            int maxit = opts.maxit;
            
            int n = A.size();
            double res = {0.0};
            int nit = maxit;
            bool isconverged = false;
            std::vector<double> history;

            SolverResult info = {res, nit, isconverged, history};

            std::vector<double> x_new(n);
            double b_nrm = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
            for (int it = 0; it < maxit; ++it) {
                for (int i = 0; i < n; ++i) {
                    double num = 0.0;
                    for (int j = 0; j < n; ++j) {
                        if (i != j) num += A[i][j] * x[j];
                    }
                    if (std::abs(A[i][i]) <= 1e-16) {
                        std::cerr << "Diagonal matrix contains 0!\n";
                        info.isconverged = false;
                        return info;
                    }
                    x_new[i] = (-num + b[i]) / A[i][i];
                }
                
                swap(x, x_new);

                double r = resNrm(A, b, x);
                info.history.push_back(std::abs(r / b_nrm));
                
                if (std::abs(b_nrm) > 1e-16 && r / b_nrm <= reltol) {
                    info.isconverged = true;
                    info.it = it;
                    info.res = r;
                    return info;
                }
            }

            return info;
        }

inline SolverResult GaussSeidel(const std::vector<std::vector<double>>& A, const std::vector<double>& b, 
    std::vector<double>& x, const SolverOptions& opts) {
        double reltol = opts.reltol;
        int maxit = opts.maxit;
        
        int n = A.size();
        std::vector<double> history;
        SolverResult info = {0.0, 0, false, history};

        std::vector<double> x_new(n, 0.0);
        double b_nrm = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
        for (int it = 0; it < maxit; ++it) {
            for (int i = 0; i < n; ++i) {
                double num = 0.0;
                for (int j = 0; j <= i - 1; ++j) {
                    num += A[i][j] * x_new[j];
                }

                for (int j = i + 1; j < n; ++j) {
                    num += A[i][j] * x[j];
                }

                if (std::abs(A[i][i]) < 1e-16) {
                    std::cerr << "Diagonal value almost 0!\n";
                    info.isconverged = false;
                    return info;
                }

                x_new[i] = (b[i] - num) / A[i][i];

            }
            swap(x, x_new);

            double r = resNrm(A, b, x);
            info.history.push_back(std::abs(r / b_nrm));

            if (std::abs(b_nrm) > 1e-16 && std::abs(r / b_nrm) <= reltol) {
                info.isconverged = true;
                info.it = it;
                info.res = r;
                return info;
            }

        }

        return info;
}

inline SolverResult GMRES(const std::vector<std::vector<double>>& A, const std::vector<double>& b, 
    std::vector<double>& x, const SolverOptions& opts) {
    // need to think about this agian, Km subspace and Hessenberg matrix
    // QR factorization ? Given rotations? Restart?
}