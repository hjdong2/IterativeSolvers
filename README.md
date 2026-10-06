
# IterativeSolvers

IterativeSolvers is a C++20 library implementing classical iterative solvers and Krylov methods from scratch, using Eigen for the matrix types, and tested against Eigen's direct solvers.

# Features

- Jacobi, Gauss-Seidel
- Classical and modified Gram-Schmidt orthogonalization
- Arnoldi algorithm (MGS-based, with breakdown)
- GMRES with Givens Rotations

# Results

## CGS vs. MGS loss of orthogonality using Hilbert matrices

Produced by `gram_schmidt_stability`
QR factorization of Hilbert matrices H(i, j) = 1 / (i + j + 1), which 
becomes extremely ill-conditioned as matrix size grows. The table
shows ||I - Q^TQ||_F:

|    n    |     CGS    |    MGS    |
|---------|------------|-----------|
|    4    |   6.2e-11  |   4.2e-13 |
|    6    |   2.4e-04  |   6.8e-10 |
|    8    |   1.4e+00  |   2.3e-07 |
|    10   |   3.4e+00  |   1.6e-04 |
|    12   |   5.5e+00  |   1.1e-01 |

CGS lost orthogonality: by n = 8, 
CGS returns a Q that is not orthogonal at all, while 
MGS can still give 1e-7.
This is why the Arnoldi algorithm used MGS.

## Jacobi vs. Gauss-Seidel on 10-by-10 tridiagonal matrix

Produced by `jacobi_gs_demo`
The matrix: 10x10 tridiagonal: 4 on the diagonal, -1 on the off-diagonals, 
b = (1, 2, ..., 10), and x_0 = 0. 

|    Method    |    Iterations    |    Final relative residual    |      x - x_exact      |
|--------------|------------------|-------------------------------|-----------------------|
|    Jacobi    |      25          |         8.8e-09               |      8.3e-08          |
| Gauss-Seidel |      15          |         9.0e-09               |      7.0e-08          |


- Both methods have the form x_{k+1} = G x_k + c.

## GMRES with Givens Rotations

Produced by `gmres_demo`. The same 10x10 tridiagonal matrix, b = (1, 2, ..., 10), x_0 = 0. Each row
is a separate GMRES run with Krylov dimension m = k.


|    Step k        |    Givens residual    |    True residual     |
|------------------|-----------------------|----------------------|
|       1          |      2.06e-01         |      2.06e-01        |
|       2          |      5.12e-02         |      5.12e-02        |
|       3          |      1.32e-02         |      1.32e-02        |
|       5          |      8.56e-04         |      8.56e-04        |
|       8          |      1.12e-05         |      1.12e-05        |
|       9          |      2.04e-06         |      2.04e-06        |
|       10         |      2.70e-16         |      5.70e-16        |

- GMRES finds the x with the smallest residual in the Krylov subpsace by solving a small least-square problem with 
Hessenberg matrix H.
- Givens rotations zero out the subdiagonal values of Hessenberg matrix one column at a time. The last entry of 
the rotated right-hand side is the residual norm, so convergence can be monitored at every step without computing
|| b - A x ||, which would cost an extra matrix-vector product.
- The residual at the final iteration differ. Both values are at  near machine precision (~1e-16), so the difference 
is only rounding.

# Build and run

## Requirements
- A C++20 compiler (tested with GCC 16 via MSYS2 UCRT64 on Windows)
- CMake >= 3.20
- Eigen 3 (header-only)
- Ninja (optional; any CMake generator works)

Installing the dependencies on Windows (MSYS2 UCRT64 terminal):

```bash
pacman -S mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-eigen3
```

## Build

```bash
git clone https://github.com/hjdong2/IterativeSolvers.git
cd IterativeSolvers
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run the tests

```bash
ctest --test-dir build --output-on-failure
```

## Run the examples

```bash
./build/gram_schmidt_stability   # CGS vs. MGS loss of orthogonality on Hilbert matrices
./build/jacobi_gs_demo           # Jacobi vs. Gauss-Seidel on a tridiagonal system
./build/gmres_demo               # GMRES with Givens rotations  
```

On Windows (PowerShell), use `.\build\gram_schmidt_stability.exe`, etc.

# Possible future extensions

- GMRES(m) with Restart
- Left / right preconditioning
