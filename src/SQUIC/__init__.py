"""
SQUIC
================

description :
SQUIC is a second-order, L1-regularized maximum likelihood method for performant large-scale sparse precision matrix estimation.


usage :
 SQUIC(Y, lambda, max_iter = 100, tol = 1e-3, verbose = 1, M = NULL,  X0 = NULL, W0 = NULL)

arguments :
    Y:          Input data in the form p (dimensions) by n (samples).
    l:          (Non-zero positive paramater) Scalar tuning parameter controlling the sparsity of the precision matrix.
    max_iter:   Maximum number of Newton iterations of outer loop. Default: 100.
    tol:        Tolerance for convergence and approximate inversion. Default: 1e-3.
    verbose:    Level of printing output (0 or 1). Default: 1.
    M:          The matrix encoding the sparsity pattern of the matrix tuning parameter, i.e., Lambda (p by p). Default: NULL.
    X0:         Initial guess of precision matrix (p by p). Default: NULL.
    W0:         Initial guess of the inverse of the precision matrix (p by p). Default: NULL.


details :
SQUIC is a high-performance precision matrix estimation package intended for large-scale applications. The algorithm utilizes second-order information in conjunction with a highly optimized, supernodal Cholesky factorization routine (CHOLMOD), a block-oriented computational approach for both the coordinate descent and the approximate matrix inversion, and finally, a sparse representation of the sample covariance matrix. The library is implemented in C++ with shared-memory parallelization using OpenMP.
For further details, please see the listed references.


return values :
 A list with components:

 [X,W,info_times,info_objective,info_logdetX,info_trSX]

 X: Estimated precision matrix (p by p).
 W: Estimated inverse of the precision matrix (p by p).
 info_times: List of different compute times.
    info_time_total: Total runtime.
    info_time_sample_cov: Runtime of the sample covariance matrix.
    info_time_optimize: Runtime of the Newton steps.
    info_time_factor: Runtime of the Cholesky factorization.
    info_time_approximate_inv: Runtime of the approximate matrix inversion.
    info_time_coordinate_upd: Runtime of the coordinate descent update.
 info_objective: Value of the objective at each Newton iteration.
 info_logdetX: Value of the log determinant of the precision matrix.
 info_trSX: Value of the trace of the sample covariance matrix times the precision matrix.


references :
Bollhöfer, M., Eftekhari, A., Scheidegger, S. and Schenk, O., 2019. Large-scale sparse inverse covariance matrix estimation.
SIAM Journal on Scientific Computing, 41(1), pp.A380-A401.

Eftekhari, A., Bollhöfer, M. and Schenk, O., 2018, November. Distributed memory sparse inverse covariance matrix estimation
on high-performance computing architectures.
In SC18: International Conference for High Performance Computing, Networking, Storage and Analysis (pp. 253-264). IEEE.

Eftekhari, A., Pasadakis, D., Bollhöfer, M., Scheidegger, S. and Schenk, O., 2021. Block-Enhanced Precision Matrix Estimation
for Large-Scale Datasets. Journal of Computational Science, p. 101389.


example :

import SQUIC
import numpy as np

# generate sample from tridiagonal precision matrix
p = 1024
n = 100
l = .4
max_iter = 100
tol = 1e-3

# generate a tridiagonal matrix
a = -0.5 * np.ones(p-1)
b = 1.25 * np.ones(p)
iC_star = np.diag(a,-1) + np.diag(b,0) + np.diag(a,1)

# generate the data
L = np.linalg.cholesky(iC_star)
Y = np.linalg.solve(L.T,np.random.randn(p,n))

[X,W,info_times,info_objective,info_logdetX,info_trSX] = SQUIC.run(Y,l)

"""

from .SQUIC_Python import set_path

from .SQUIC_Python import run
from .SQUIC_Python import S_run

