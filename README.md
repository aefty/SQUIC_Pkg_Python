## SQUIC for Python
### Sparse Quadratic Inverse Covariance Estimation
This is the SQUIC algorithm, made available as a Python package. 
SQUIC tackles the statistical problem of estimating large sparse 
inverse covariance matrices. This estimation poses an ubiquitous 
problem that arises in many applications e.g. coming from the 
fields mathematical finance, geology and health. 
SQUIC belongs to the class of second-order L1-regularized 
Gaussian maximum likelihood methods and is especially suitable 
for high-dimensional datasets with limited number of samples. 
For further details please see the listed references.

### References

[1] Eftekhari A, Pasadakis D, Bollhöfer M, Scheidegger S, Schenk O (2021). Block-Enhanced PrecisionMatrix Estimation for Large-Scale Datasets.Journal of Computational Science, p. 101389.

[2] Bollhöfer, M., Eftekhari, A., Scheidegger, S. and Schenk, O., 2019. Large-scale sparse inverse covariance matrix estimation. SIAM Journal on Scientific Computing, 41(1), pp.A380-A401.

[3] Eftekhari, A., Bollhöfer, M. and Schenk, O., 2018, November. Distributed memory sparse inverse covariance matrix estimation on high-performance computing architectures. In SC18: International Conference for High Performance Computing, Networking, Storage and Analysis (pp. 253-264). IEEE.

### Installation

We are currently supporting Linux and MacOS distributions.

Before installing the SQUIC for Python package, the pre-compiled library 
libSQUIC(.dylib/.so) needs to be installed from https://www.gitlab.ci.inf.usi.ch/SQUIC/libSQUIC. Please follow the instructions provided there.

Now you can install the SQUIC for Python package from PyPI using 

```angular2
pip install SQUIC
```

The libSQUIC(.dylib/.so) file is by default be installed in your home directory. 
This way the SQUIC package for Python will find it automatically. Otherwise you can manually 
set the path using

```angular2
import SQUIC
SQUIC.set_path("/path/to/libSQUIC(.dylib/.so)")
```

### Example I 

```angular2
import SQUIC 
import numpy as np

# number of covariates p, number of samples n
p=1024
n=100

# set regularisation parameter lambda
l=0.4

# sample synthetic data from standard normal distribution 
Y=np.random.randn(p,n)

# compute sparse precision matrix and its inverse
[X,W,info_times,info_objective,info_logdetX,info_trSX]= SQUIC.run(Y=Y,l=l)
```

### Example II

```angular2
# set OMP_NUM_THREADS before loading SQUIC
import os
os.environ["OMP_NUM_THREADS"] = '4'

# load SQUIC and numpy
import SQUIC 
import numpy as np

# generate sample from tridiagonal precision matrix
# number of covariates p
p = 1024
# number of samples n
n = 100

# define tridiagonal precision matrix iC_star
a = -0.5 * np.ones(p-1)
b = 1.25 * np.ones(p)
iC_star = np.diag(a,-1) + np.diag(b,0) + np.diag(a,1)

# compute Cholesky factorisation
L = np.linalg.cholesky(iC_star)

# sample from standard normal distribution & solve
# Y contains n samples from multivariate Gaussian distribution
# with zero mean and precision matrix iC_star
Y = np.linalg.solve(L.T,np.random.randn(p,n))

# set regularisation parameter lambda
l = 0.3
# compute sample covariance matrix
[S,info_times] = SQUIC.S_run(Y, l)

# compute sparse precision matrix and its inverse
[X,W,info_times,info_objective,info_logdetX,info_trSX] = SQUIC.run(Y,l)

# X contains estimate for iC_star
# W contains inverse of X
```

For a detailed list of all (optional) input and output parameters: 

```angular2
help(SQUIC.S_run)
help(SQUIC.run)
```
