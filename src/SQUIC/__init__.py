"""
#################
# Description:
#################

SQUIC is a second-order, L1-regularized maximum likelihood method for performant large-scale sparse precision matrix estimation.

#################
# Usage :
#################

[X,W,info_times,info_objective,info_logdetX,info_trSX]=SQUIC.run(Y, lambda, max_iter = 100, tol = 1e-3, verbose = 1, M = NULL,  X0 = NULL, W0 = NULL)

Arguments:
    Y:          Input data in the form p (dimensions) by n (samples).
    l:          Scalar tuning parameter controlling the sparsity of the precision matrix (a non-zero positive number).
    max_iter:   Maximum number of Newton iterations of the outer loop. Default: 100.
    tol:        Tolerance for convergence and approximate inversion. Default: 1e-3.
    verbose:    Level of printing output (0 or 1). Default: 1.
    M:          The matrix encoding the sparsity pattern of the matrix tuning parameter, i.e., Lambda (p by p). Default: NULL.
    X0:         Initial guess of the precision matrix (p by p). Default: Identity.
    W0:         Initial guess of the inverse of the precision matrix (p by p). Default: Identity.

NOTE:
(1) If max_iter=0, the returned value for the inverse of the precision matrix (W) is the sparse sample covariance matrix (S).
(2) The number of threads used by SQUIC can be defined by setting the enviroment variable OMP_NUM_THREADS (e.g., base> export OMP_NUM_THREADS=12). This may require a restart of the session).


Return values:

 X: Estimated precision matrix (p by p).
 W: Estimated inverse of the precision matrix (p by p).
 info_times: List of different compute times.
    [0] info_time_total: Total runtime.
    [1] info_time_sample_cov: Runtime of the sample covariance matrix.
    [2] info_time_optimize: Runtime of the Newton steps.
    [3] info_time_factor: Runtime of the Cholesky factorization.
    [4] info_time_approximate_inv: Runtime of the approximate matrix inversion.
    [5] info_time_coordinate_upd: Runtime of the coordinate descent update.
 info_objective: Value of the objective at each Newton iteration.
 info_logdetX: Value of the log determinant of the precision matrix.
 info_trSX: Value of the trace of the sample covariance matrix times the precision matrix.

#################
# References :
#################

Bollhöfer, M., Eftekhari, A., Scheidegger, S. and Schenk, O., 2019. Large-Scale Sparse Inverse Covariance Matrix Estimation.
SIAM Journal on Scientific Computing, 41(1), pp.A380-A401.

Eftekhari, A., Bollhöfer, M. and Schenk, O., 2018, November. Distributed Memory Sparse Inverse Covariance Matrix Estimation
on High-performance Computing Architectures.
In SC18: International Conference for High Performance Computing, Networking, Storage and Analysis (pp. 253-264). IEEE.

Eftekhari, A., Pasadakis, D., Bollhöfer, M., Scheidegger, S. and Schenk, O., 2021. Block-Enhanced Precision Matrix Estimation
for Large-Scale Datasets. Journal of Computational Science, p. 101389.

#################
# Example:
#################

import SQUIC 
import numpy as np

# set location of libSQUIC (set after importing package)
SQUIC.PATH_TO_libSQUIC('/path/to/squic/')

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

from .SQUIC_Python import PATH_TO_libSQUIC
from .SQUIC_Python import run

