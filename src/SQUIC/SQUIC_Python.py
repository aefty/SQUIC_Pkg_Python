from ctypes import *
import numpy as np
from scipy.sparse import csr_matrix, identity
from pathlib import Path

dll = None

def set_path(libSQUIC_path):

    global shared_lib_path
    global dll

    shared_lib_path = libSQUIC_path

    try:
        dll = CDLL(shared_lib_path)
        print("libSQUIC Successfully loaded!", dll)
        return True
    except Exception:
        print("libSQUIC could not be found under the current path.")
        return False

def check_path():

    if dll != None:
        if os.path.exists(dll._name):
            print(dll)
            print("libSQUIC already loaded.")
            return True
        else:
            print(f"libSQUIC not found : {dll}")
            return False
    else:
        home_dir = str(Path.home())
        file_name_dylib = "/libSQUIC.dylib"
        file_name_so = "/libSQUIC.so"

        if set_path(home_dir + file_name_dylib) or set_path(home_dir + file_name_so):

            return True
        else:
            print("libSQUIC path not set correctly or library not yet downloaded. Add path by calling: SQUIC.set_path(libSQUIC_path).")
            return False


def run(Y, l, max_iter=100, tol=1e-3,verbose=1, M=None, X0=None, W0=None):
    """
    Sparse Inverse Covariance Estimation
    :param Y: Input data in the form p (dimensions) by n (samples).
    :param l: (Non-zero positive paramater) Scalar tuning parameter controlling the sparsity of the precision matrix.
    :param max_iter: Maximum number of Newton iterations of outer loop. Default: 100.
    :param tol: Tolerance for convergence and approximate inversion. Default: 1e-3.
    :param verbose: Level of printing output (0 or 1). Default: 1.
    :param M: The matrix encoding the sparsity pattern of the matrix tuning parameter, i.e., Lambda (p by p). Default: NULL.
    :param X0: Initial guess of precision matrix (p by p). Default: NULL.
    :param W0: Initial guess of the inverse of the precision matrix (p by p). Default: NULL.
    :return: [X,W,info_times,info_objective,info_logdetX,info_trSX]
        :return: X: Estimated precision matrix (p by p).
        :return: W: Estimated inverse of the precision matrix (p by p).
        :return: info_times: List of different compute times.
            :return: info_time_total: Total runtime.
            :return: info_time_sample_cov: Runtime of the sample covariance matrix.
            :return: info_time_optimize: Runtime of the Newton steps.
            :return: info_time_factor: Runtime of the Cholesky factorization.
            :return: info_time_approximate_inv: Runtime of the approximate matrix inversion.
            :return: info_time_coordinate_upd: Runtime of the coordinate descent update.
        :return: info_objective: Value of the objective at each Newton iteration.
        :return: info_logdetX: Value of the log determinant of the precision matrix.
        :return: info_trSX: Value of the trace of the sample covariance matrix times the precision matrix.
    """

    if not check_path():
        return

    # if mode = [0,1,2,3,4] we Block-SQUIC or [5,6,7,8,9] Scalar-SQUIC
    mode = c_int(0)

    p,n= Y.shape

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

    # X Matrix
    if(X0==None):
        X0= identity(p, dtype='float64', format='csr')

    X_rinx = POINTER(c_long)()
    X_cptr = POINTER(c_long)()
    X_val  = POINTER(c_double)()
    X_nnz  = c_long(X0.nnz)

    # Use the SQUIC_CPP utility functino for creating and populating data buffers.
    # This makes creating and delete buffers consistent.
    # All SQUIC datastructions are 64 bit! Thus we need to cast the CSR index buffers to int64
    dll.SQUIC_CPP_UTIL_memcopy_integer(byref(X_rinx) , np.int64(X0.indices).ctypes.data_as(POINTER(c_long))  , X_nnz       )
    dll.SQUIC_CPP_UTIL_memcopy_integer(byref(X_cptr) , np.int64(X0.indptr).ctypes.data_as(POINTER(c_long))   , c_long(p+1) )
    dll.SQUIC_CPP_UTIL_memcopy_double( byref(X_val)  , X0.data.ctypes.data_as(POINTER(c_double))             , X_nnz       )

    # W Matrix
    if(W0==None):
        W0= identity(p, dtype='float64', format='csr')

    W_rinx = POINTER(c_long)()
    W_cptr = POINTER(c_long)()
    W_val  = POINTER(c_double)()
    W_nnz  = c_long(W0.nnz)

    dll.SQUIC_CPP_UTIL_memcopy_integer(byref(W_rinx) , np.int64(W0.indices).ctypes.data_as(POINTER(c_long))  , W_nnz       )
    dll.SQUIC_CPP_UTIL_memcopy_integer(byref(W_cptr) , np.int64(W0.indptr).ctypes.data_as(POINTER(c_long))   , c_long(p+1) )
    dll.SQUIC_CPP_UTIL_memcopy_double( byref(W_val)  , W0.data.ctypes.data_as(POINTER(c_double))             , W_nnz       )

    # M Matrix
    M_rinx = POINTER(c_long)()
    M_cptr = POINTER(c_long)()
    M_val  = POINTER(c_double)()

    if(M==None):
        M_nnz  = c_long(0)
    else:
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
        byref(info_times_ptr),	   # length must be 6: [time_total,time_impcov,time_optimz,time_factor,time_aprinv,time_updte]
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

    X_rinx_py	=np.ctypeslib.as_array(X_rinx,(X_nnz,))
    X_cptr_py	=np.ctypeslib.as_array(X_cptr,(p+1,))
    X_val_py	=np.ctypeslib.as_array(X_val,(X_nnz,))
    X 			=csr_matrix((X_val_py, X_rinx_py, X_cptr_py),shape=(p, p),copy=True)

    W_rinx_py	=np.ctypeslib.as_array(W_rinx,(W_nnz,))
    W_cptr_py	=np.ctypeslib.as_array(W_cptr,(p+1,))
    W_val_py	=np.ctypeslib.as_array(W_val,(W_nnz,))
    W 			=csr_matrix((W_val_py, W_rinx_py, W_cptr_py),shape=(p, p),copy=True)

    temp			=np.ctypeslib.as_array(info_objective_ptr,(info_num_iter,))
    info_objective	=np.array(temp,copy=True)

    temp			=np.ctypeslib.as_array(info_times_ptr,(6,))
    info_times   	=np.array(temp,copy=True)

    #################################################
    # Transfer Restusl from C to Python
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

    dll.SQUIC_CPP_UTIL_memfree_double(byref(info_objective_ptr))
    dll.SQUIC_CPP_UTIL_memfree_double(byref(info_times_ptr))

    return [X,W,info_times,info_objective,info_logdetX,info_trSX]


def S_run(Y, l,verbose=1, M=None):
    """
    :param Y: Input data in the form p (dimensions) by n (samples).
    :param l: (Non-zero positive paramater) Scalar tuning parameter controlling the sparsity of the precision matrix.
    :param verbose: Level of printing output (0 or 1). Default: 1.
    :param M: The matrix encoding the sparsity pattern of the matrix tuning parameter, i.e., Lambda (p by p). Default: NULL.
    :return: [S, info_times]
        :return: S: sparse(thresholded) sample covariance matrix.
        :return: info_times: List of different compute times.
            :return: info_time_total: Total compute time of the S_run function call.
            :return: info_time_sample_cov: Compute time of the sample covariance matrix.
    """

    [_,S,info_times,_,_,_]=run(Y=Y, l=l, max_iter=0, verbose=verbose, M=M)

    return [S,info_times]


