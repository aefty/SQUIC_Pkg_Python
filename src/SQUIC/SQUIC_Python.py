from ctypes import *
import numpy as np
import os
import sys
from scipy.sparse import csr_matrix, identity
from pathlib import Path

dll = None

def PATH_TO_libSQUIC(libSQUIC_path):
    """
#################
# Description:
#################    
Set the path of libSQUIC. Require only once after importing.
See help(SQUIC) for further details.

#################
# Usage :
#################
Arguments:
    libSQUIC_path:  path to libSQUIC; e.g., /User/bob/ (note the slash at the end).

    """    

    global dll

    libSQUIC_loc = libSQUIC_path
    if sys.platform.startswith('darwin'):
        libSQUIC_loc=libSQUIC_loc+"libSQUIC.dylib"
    elif sys.platform.startswith('linux'):
        libSQUIC_loc=libSQUIC_loc+"libSQUIC.so"
    else:
        raise Exception("#SQUIC: OS not supported.");

    dll = CDLL(libSQUIC_loc)
    print( libSQUIC_loc + " loaded!", dll)


def run(Y, l, max_iter=100, tol=1e-3,verbose=1, M=None, X0=None, W0=None):
    """ 
#################
# Description:
#################
SQUIC is a second-order, L1-regularized maximum likelihood method for performant large-scale sparse precision matrix estimation.
See help(SQUIC) for further details.

#################
# Usage :
#################
Arguments:
    Y:          Input data in the form p (dimensions) by n (samples).
    l:          (Non-zero positive parameter) Scalar tuning parameter controlling the sparsity of the precision matrix.
    max_iter:   Maximum number of Newton iterations of the outer loop. Default: 100.
    tol:        Tolerance for convergence and approximate inversion. Default: 1e-3.
    verbose:    Level of printing output (0 or 1). Default: 1.
    M:          The matrix encoding the sparsity pattern of the matrix tuning parameter, i.e., Lambda (p by p). Default: NULL.
    X0:         Initial guess of the precision matrix (p by p). Default: NULL.
    W0:        Initial guess of the inverse of the precision matrix (p by p). Default: NULL.

Note: If max_iter=0, the returned value for the inverse of the precision matrix (W) is the sparse sample covariance matrix (S).

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
    """

    if(dll==None):
        raise Exception("#SQUIC: libSQUIC not loaded, use SQUIC.PATH_TO_libSQUIC(libSQUIC_path).");

    p,n= Y.shape

    if(p<3):
        raise Exception("#SQUIC: number of random variables (p) must larger than 2");

    if(n<2):
        raise Exception("#SQUIC: number of samples (n) must be larger than 1 .");

    if(l<=0):
        raise Exception("#SQUIC: lambda must be great than zero.");

    if(max_iter<0):
        raise Exception("#SQUIC: max_iter cannot be negative.");

    if(tol<=0):
        raise Exception("#SQUIC: tol must be great than zero.");



    #################################################
    # if mode = [0,1,2,3,4] we Block-SQUIC or [5,6,7,8,9] Scalar-SQUIC
    mode = c_int(0)

    # The data needs to be fortran (column major)
    Y=np.array(Y,order='F')
    Y_ptr   = Y.ctypes.data_as(POINTER(c_double))

    #################################################
    # tolerances
    # Hard code both tolerances to be the same
    term_tol = tol
    inv_tol = tol

    #################################################
    # Matrices
    #################################################

    # X & W Matrix Checks
    if(X0==None or W0==None ): 
        # Make identity sparse matrix.
        X0= identity(p, dtype='float64', format='csr')
        W0= identity(p, dtype='float64', format='csr')
    else:

        # Check size
        [X0_p,X0_n]=X0.shape
        if(X0_p!=p or X0_p!=p ):
            raise Exception("#SQUIC: X0 must be square matrix with size pxp..")

        # Check size
        [W0_p,W0_n]=W0.shape
        if(W0_p!=p or W0_p!=p ):
            raise Exception("#SQUIC: W0 must be square matrix with size pxp..")    

        # Force Symmetric
        X0=(X0+X0.T)/2;
        W0=(W0+W0.T)/2;


    #################################################
    # Allocate data fo X
    #################################################
    X_rinx = POINTER(c_long)()
    X_cptr = POINTER(c_long)()
    X_val  = POINTER(c_double)()
    X_nnz  = c_long(X0.nnz)

    # Use the SQUIC_CPP utility function for creating and populating data buffers.
    # This makes creating and delete buffers consistent.
    # All SQUIC datastructions are 64 bit! Thus we need to cast the CSR index buffers to int64
    dll.SQUIC_CPP_UTIL_memcopy_integer(byref(X_rinx) , np.int64(X0.indices).ctypes.data_as(POINTER(c_long))  , X_nnz       )
    dll.SQUIC_CPP_UTIL_memcopy_integer(byref(X_cptr) , np.int64(X0.indptr).ctypes.data_as(POINTER(c_long))   , c_long(p+1) )
    dll.SQUIC_CPP_UTIL_memcopy_double( byref(X_val)  , X0.data.ctypes.data_as(POINTER(c_double))             , X_nnz       )

    #################################################
    # Allocate data fo W
    #################################################
    W_rinx = POINTER(c_long)()
    W_cptr = POINTER(c_long)()
    W_val  = POINTER(c_double)()
    W_nnz  = c_long(W0.nnz)

    dll.SQUIC_CPP_UTIL_memcopy_integer(byref(W_rinx) , np.int64(W0.indices).ctypes.data_as(POINTER(c_long))  , W_nnz       )
    dll.SQUIC_CPP_UTIL_memcopy_integer(byref(W_cptr) , np.int64(W0.indptr).ctypes.data_as(POINTER(c_long))   , c_long(p+1) )
    dll.SQUIC_CPP_UTIL_memcopy_double( byref(W_val)  , W0.data.ctypes.data_as(POINTER(c_double))             , W_nnz       )



    #################################################
    # Check and Allocated data for M
    #################################################
    M_rinx = POINTER(c_long)()
    M_cptr = POINTER(c_long)()
    M_val  = POINTER(c_double)()

    if(M==None):
        M_nnz  = c_long(0)
    else:
        
        # Check size
        [M_p,M_n]=M.shape
        if(M_p!=p or M_n!=p ):
            raise Exception("#SQUIC: M must be square matrix with size pxp..")    

        # Make all postive, drop all zeros and force symmetrix
        M = eliminate_zeros(M);
        M = absolute(M);
        M = (M + M)/2; 

        M_nnz  = c_long(M.nnz)
        dll.SQUIC_CPP_UTIL_memcopy_integer(byref(M_rinx) , np.int64(M.indices).ctypes.data_as(POINTER(c_long))  , M_nnz       )
        dll.SQUIC_CPP_UTIL_memcopy_integer(byref(M_cptr) , np.int64(M.indptr).ctypes.data_as(POINTER(c_long))   , c_long(p+1) )
        dll.SQUIC_CPP_UTIL_memcopy_double( byref(M_val)  , M.data.ctypes.data_as(POINTER(c_double))             , M_nnz       )


    #################################################
    # Parameters
    #################################################
    max_iter_ptr  = c_int(max_iter);
    term_tol_ptr  = c_double(term_tol);
    inv_tol_ptr  = c_double(inv_tol);
    verbose_ptr   = c_int(verbose)

    p_ptr    = c_long(p)
    n_ptr    = c_long(n)
    l_ptr    = c_double(l)

    #################################################
    # Information output buffers
    #################################################
    info_num_iter_ptr    = c_int(-1);
    info_logdetX_ptr     = c_double(-1);
    info_trSX_ptr        = c_double(-1);

    info_times_ptr = POINTER(c_double)()
    dll.SQUIC_CPP_UTIL_memset_double(byref(info_times_ptr),c_int(6))

    info_objective_ptr = POINTER(c_double)()
    dll.SQUIC_CPP_UTIL_memset_double(byref(info_objective_ptr),max_iter_ptr)

    #################################################
    # Run SQUIC
    #################################################
    dll.SQUIC_CPP(mode,
        p_ptr, n_ptr, Y_ptr,
        l_ptr,
        M_rinx, M_cptr, M_val, M_nnz,
        max_iter_ptr, inv_tol_ptr, term_tol_ptr, verbose_ptr,
        byref(X_rinx), byref(X_cptr), byref(X_val), byref(X_nnz),
        byref(W_rinx), byref(W_cptr), byref(W_val), byref(W_nnz),
        byref(info_num_iter_ptr),
        byref(info_times_ptr),     # length must be 6: [time_total,time_impcov,time_optimz,time_factor,time_aprinv,time_updte]
        byref(info_objective_ptr), # length must be size max_iter
        byref(info_logdetX_ptr),
        byref(info_trSX_ptr))


    #################################################
    # Transfer Restusl from C to Python
    #################################################

    #Convert all scalars values back python
    p               = p_ptr.value
    X_nnz           = X_nnz.value
    W_nnz           = W_nnz.value
    info_num_iter   = info_num_iter_ptr.value
    info_logdetX    = info_logdetX_ptr.value
    info_trSX       = info_trSX_ptr.value

    # Transfer Matrix from C to Python
    # First we link the C buffer to a numpy array, than we make CSR matrix using a Copy of the array.

    X_rinx_py   =np.ctypeslib.as_array(X_rinx,(X_nnz,))
    X_cptr_py   =np.ctypeslib.as_array(X_cptr,(p+1,))
    X_val_py    =np.ctypeslib.as_array(X_val,(X_nnz,))
    X           =csr_matrix((X_val_py, X_rinx_py, X_cptr_py),shape=(p, p),copy=True)

    W_rinx_py   =np.ctypeslib.as_array(W_rinx,(W_nnz,))
    W_cptr_py   =np.ctypeslib.as_array(W_cptr,(p+1,))
    W_val_py    =np.ctypeslib.as_array(W_val,(W_nnz,))
    W           =csr_matrix((W_val_py, W_rinx_py, W_cptr_py),shape=(p, p),copy=True)

    if(info_num_iter==0):
        info_objective=np.array(-1)
    else:
        temp            =np.ctypeslib.as_array(info_objective_ptr,(info_num_iter,))
        info_objective  =np.array(temp,copy=True)

    temp            =np.ctypeslib.as_array(info_times_ptr,(6,))
    info_times      =np.array(temp,copy=True)

    #################################################
    # Transfer results from C to Python
    #################################################
    dll.SQUIC_CPP_UTIL_memfree_integer(byref(X_rinx))
    dll.SQUIC_CPP_UTIL_memfree_integer(byref(X_cptr))
    dll.SQUIC_CPP_UTIL_memfree_double(byref(X_val))

    dll.SQUIC_CPP_UTIL_memfree_integer(byref(W_rinx))
    dll.SQUIC_CPP_UTIL_memfree_integer(byref(W_cptr))
    dll.SQUIC_CPP_UTIL_memfree_double(byref(W_val))

    if(M!=None):
        dll.SQUIC_CPP_UTIL_memfree_integer(byref(M_rinx))
        dll.SQUIC_CPP_UTIL_memfree_integer(byref(M_cptr))
        dll.SQUIC_CPP_UTIL_memfree_double(byref(M_val))

    if(info_num_iter>0):    
        dll.SQUIC_CPP_UTIL_memfree_double(byref(info_objective_ptr))

    dll.SQUIC_CPP_UTIL_memfree_double(byref(info_times_ptr))

    return [X,W,info_times,info_objective,info_logdetX,info_trSX]
