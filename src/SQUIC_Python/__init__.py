"""
SQUIC_Python :
====
Package for Sparse Quadratic Inverse Covariance Estimation.

----------
package needs to be linked to libSQUIC library available under https://github.com/aefty/SQUIC_Release_Source.

check available function calls:

import SQUIC_Python as sq
help(sq.SQUIC)
help(sq.SQUIC_S)

or run tests using

import SQUIC_Python.test

SQUIC_Python.test.run_SQUIC()
SQUIC_Python.test.run_SQUIC_S()


"""

from .SQUIC_Python import set_path
from .SQUIC_Python import get_path
from .SQUIC_Python import check_path

from .SQUIC_Python import SQUIC
from .SQUIC_Python import SQUIC_S
