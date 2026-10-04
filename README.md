
# IterativeSolvers
IterativeSolvers is a C++20 library impementing classical iterative solvers and Krylov methods from scratch, using Eigen for the matrix types, and tested against Eigen's direct solvers.

# Features
- Jacobi, Gauss-Seidel
- Classical and modified Gram-Schmidt projection
- Arnoldi algorithm (MGS-based, with breakdown)
- GMRES with Given Rotations

# Results
- CGS vs. MGS loss of orthogonality using Hilbert matrices
- Jacobi vs. Gauss-Seidel on 10-by-10 tridiagonal matrix
- 3-by-3 matrix where Jacobi diverged but Gauss-Seidel converges
- GMRES with Given Rotations

# Possible future extensions
- GMRES(m) with Restart
- Left / right preconditioning
