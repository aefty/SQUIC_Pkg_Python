from ctypes import *
from sys import platform
import numpy as np
from scipy.sparse import csr_matrix,isspmatrix_csr, identity


shared_lib_path = "/path/to/libSQUIC"
dll = None


def set_path(libSQUIC_path):

	global shared_lib_path 
	global dll

	shared_lib_path = libSQUIC_path

	try:
		dll = CDLL(shared_lib_path)
		print("libSQUIC Successfully loaded!", dll)
		return True
	except Exception as e:
		print(e)
		return False


def run(Y_learn,Y_test,l,M,W0,X0,max_iter,term_tol,drop_tol,verbose):	

	# if mode = [0,1,2,3,4] we Block-SQUIC or [5,6,7,8,9] Scalar-SQUIC
	mode = c_int(0)

	p,n1= Y_learn.shape

	# The data needs to be fortran (column major)
	Y_learn=np.array(Y_learn,order='F')
	Y1   = Y_learn.ctypes.data_as(POINTER(c_double))

	if(Y_test==None):
		Y2   = POINTER(c_double)()
		n2   = 0
	else:
		n2     = Y_test.shape[1]
		Y_test = np.array(Y_learn,order='F')
		Y2     = Y_test.ctypes.data_as(POINTER(c_double))
		

	#################################################
	# Matricis
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
	max_iter  = c_int(max_iter);
	term_tol  = c_double(term_tol);
	drop_tol  = c_double(drop_tol);
	verbose   = c_int(verbose)

	p    = c_long(p)
	n1   = c_long(n1)
	n2   = c_long(n2)
	l    = c_double(l)

	#################################################
	# Information output buffers
	#################################################
	info_num_iter    = c_int(-1);
	info_dgap        = c_double(-1);
	info_logdetX_Y1  = c_double(-1);
	info_trXS_Y2     = c_double(-1);

	info_times = POINTER(c_double)()
	dll.SQUIC_CPP_UTIL_memset_double(byref(info_times),c_int(6))

	info_objective = POINTER(c_double)()
	dll.SQUIC_CPP_UTIL_memset_double(byref(info_objective),max_iter)

	#################################################
	# Run SQUIC
	#################################################
	dll.SQUIC_CPP(mode,p,n1,Y1,n2,Y2, l, 
		M_rinx, M_cptr, M_val, M_nnz,
		max_iter, drop_tol, term_tol, verbose,
		byref(X_rinx),byref(X_cptr),byref(X_val),byref(X_nnz),
		byref(W_rinx), byref(W_cptr), byref(W_val), byref(W_nnz),
		byref(info_num_iter),
		byref(info_times),	   # length must be 6: [time_total,time_impcov,time_optimz,time_factor,time_aprinv,time_updte]
		byref(info_objective), # length must be size max_iter
		byref(info_dgap),
		byref(info_logdetX_Y1),
		byref(info_trXS_Y2))


	#################################################
	# Transfer Restusl from C to Python
	#################################################
	
	#Convert all scalars values back python
	p               = p.value
	X_nnz           = X_nnz.value
	W_nnz           = W_nnz.value
	info_num_iter   = info_num_iter.value
	info_dgap       = info_dgap.value
	info_logdetX_Y1 = info_logdetX_Y1.value
	info_trXS_Y2    = info_trXS_Y2.value

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

	temp			=np.ctypeslib.as_array(info_objective,(info_num_iter,))
	info_obj_trace	=np.array(temp,copy=True)

	temp			=np.ctypeslib.as_array(info_times,(6,))
	info_time_list	=np.array(temp,copy=True)


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

	dll.SQUIC_CPP_UTIL_memfree_double(byref(info_objective))
	dll.SQUIC_CPP_UTIL_memfree_double(byref(info_times))

	return [ X,W,info_obj_trace,info_time_list,info_logdetX_Y1,info_trXS_Y2,info_dgap]
