# SQUIC Python Interface Package

SQUIC is a second-order, L1-regularized maximum likelihood method for performant large-scale sparse precision matrix estimation. This repository contains the source code for the Python interface of SQUIC. 

## Installation

#### Step 1: 
Download the shared library libSQUIC from www.gitlab.ci.inf.usi.ch/SQUIC/libSQUIC, and follow its README instructions. The default and recommended location for libSQUIC is the home directory, i.e., ``~/``.

#### Step 2: 
Run the following command to install the library:
```angular2
pip install SQUIC
```
#### Step 3: 
Load the SQUIC package:
```angular2
import SQUIC
```
For further details type ``help(SQUIC)`` in the Python command line.

_Note: The number of threads used by SQUIC can be defined by setting the enviroment variable OMP_NUM_THREADS (e.g., ``base> export OMP_NUM_THREADS=12``). This may require a restart of the session)._

## Example

To run a simple example : 

```angular2
import SQUIC 
import numpy as np

# set location of libSQUIC (set after importing package)
SQUIC.PATH_TO_libSQUIC('/path/to/squic/')

# generate sample from tridiagonal precision matrix
p = 10
n = 100
l = .4

# generate a tridiagonal matrix
a = -0.5 * np.ones(p-1)
b = 1.25 * np.ones(p)
iC_star = np.diag(a,-1) + np.diag(b,0) + np.diag(a,1)

# generate the data
L = np.linalg.cholesky(iC_star)
Y = np.linalg.solve(L.T,np.random.randn(p,n))

[X,W,info_times,info_objective,info_logdetX,info_trSX] = SQUIC.run(Y,l)
```
