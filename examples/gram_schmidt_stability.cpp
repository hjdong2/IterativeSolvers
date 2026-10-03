
#include <iostream>
#include <vector>
#include <itsolvers/GramSchmidt.h>

int main() {

    std::vector<int> nums = {4, 6, 8, 10, 12};
    std::cout << "n" << "       " << "   CGS   " << "         " << "   MGS   " << '\n';
    for (auto n : nums) {
        Eigen::MatrixXd H(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                H(i, j) = 1.0 / (i + j + 1.0);
            }
        }
        // std::cout << H << '\n';
        auto [Qc, Rc] = itsolvers::classicGramSchmidt(H);
        auto [Qm, Rm] = itsolvers::modifiedGramSchmidt(H);
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
        double nrmc = (I - Qc.transpose() * Qc).norm();
        double nrmm = (I - Qm.transpose() * Qm).norm();
      
        std::cout << n << "        " << nrmc << "            " << nrmm << '\n'; 
    }

    return 0;

}