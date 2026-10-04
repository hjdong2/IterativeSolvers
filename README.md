
# IterativeSolvers
IterativeSolvers is a C++20 library impementing classical iterative solvers and Krylov methods from scratch, using Eigen for the matrix types, and tested against Eigen's direct solvers.

# Features
- Jacobi, Gauss-Seidel
- Classical and modified Gram-Schmidt projection
- Arnoldi algorithm (MGS-based, with breakdown)
- GMRES with Given Rotations

# Results
## CGS vs. MGS loss of orthogonality using Hilbert matrices
### This uses the numbers from gram_schmidt_stability demo:
### Loss of orthogonality: classical vs. modified Gram-Schmidt
QR factorizatoin of Hilbert matrices H(i, j) = 1 / (i + j + 1), which 
becomes extremely ill-conditioned as matrix size grows. The table
shows ||I - Q^TQ||_F:

|    n    |     CGS    |    MGS    |
|---------|------------|-----------|
|    4    |   6.2e-11  |   4.2e-13 |
|    6    |   2.4e-04  |   6.8e-10 |
|    8    |   1.4e+00  |   2.3e-07 |
|    10   |   3.4e+00  |   1.6e-04 |
|    12   |   5.5e+00  |   1.1e-01 |

CGS losed orthogonality: by n = 8, 
CGS returns a Q that is not orthogonal at all, while 
MGS can still give 1e-7.
This is why the Arnoldi algorithm used MGS.

## Jacobi vs. Gauss-Seidel on 10-by-10 tridiagonal matrix
## 3-by-3 matrix where Jacobi diverged but Gauss-Seidel converges
## GMRES with Given Rotations

# Build and run
## Requirements
- A C++20 compiler (tested with GCC 17 via MSYS2 UCRT64 on Windows)
- CMake >= 3.0
- Eigen3
- Ninja

### Run the examples

```bash
./build/gram_schmidt_stability   # CGS vs. MGS loss of orthogonality on Hilbert matrices
./build/jacobi_gs_demo           # Jacobi vs. Gauss-Seidel on a tridiagonal system
```

# Possible future extensions
- GMRES(m) with Restart
- Left / right preconditioning
