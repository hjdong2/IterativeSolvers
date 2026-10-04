
#pragma once
#include <iostream>
#include <vector>

namespace itsolvers {

struct SolverOptions{
    double reltol=1e-8;
    int maxit=10000;
    double w=1.0;     // for SOR relaxation
    int m=5;          // Krylov subspace dim
};

struct SolverResult{
    double res;
    int it;
    bool isconverged;
    std::vector<double> history;
};

}