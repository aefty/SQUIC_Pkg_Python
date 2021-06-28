"""
SQUIC_Python :
====
Package for Sparse Quadratic Inverse Covariance Estimation.

#### Installation libSQUIC ####

Before using SQUIC_Python for the first time, the pre-compiled library
libSQUIC needs to be installed from https://github.com/aefty/SQUIC_Release_Source.

The libSQUIC(.dylib/.so) file will by default be installed in your home directory.
This way the SQUIC_Python package will find it automatically. Otherwise you can manually
set the path using

import SQUIC_Python as SQ
SQ.set_path("/path/to/libSQUIC(.dylib/.so)")


We are currently supporting Linux and MacOS distributions.

#### Example ####

import SQUIC_Python as SQ

# generate sample from tridiagonal precision matrix
Y = SQ.generate_sample()
# set lambda = 0.5
l = 0.5
# compute sample covariance matrix
[S,info_times] = SQ.SQUIC_S(Y, l)

# compute sparse precision matrix and its inverse
[X,W,info_times,info_objective,info_logdetX,info_trSX] = SQ.SQUIC(Y,l)

where X represents the estimated sparse precision matrix and W its inverse. For a detailed list of  all (optional) input and output parameters.

help(SQ.SQUIC_S)
help(SQ.SQUIC)

"""

from .SQUIC_Python import set_path
from .SQUIC_Python import get_path
from .SQUIC_Python import check_path

from .SQUIC_Python import SQUIC
from .SQUIC_Python import SQUIC_S

