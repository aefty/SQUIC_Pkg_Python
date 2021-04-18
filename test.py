import SQUIC_Py as SQUIC
import numpy as np
from scipy.sparse import csr_matrix,isspmatrix_csr, identity


def tridiag(off_diag, diag, p):
	a=np.ones(p-1)*off_diag
	b=np.ones(p)*diag
	return np.diag(a,-1) + np.diag(b,0) + np.diag(a,1)




np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


p=10
n=8



np.random.seed(1)
iC_star = tridiag(-0.5,1.25,p)
L = np.linalg.cholesky(iC_star)
Y1 = np.linalg.solve(L.T,np.random.randn(p,n))

S=np.cov(Y1,bias=1)


print("There is n=%d samples and p=%d variables"% (n,p));
print("Sample Covariance Matrix (dont need this just for info): \n",S);
if(n>p):
	print("Inverse of Sample Covariance Matrix (dont need this just for info): \n",np.linalg.inv(S));
print("True Inverse Covariance Matrix: \n",iC_star);

# Set the SQUIC Library Path
SQUIC.set_path('/local_home/aryan/SQUIC_Release_Source/libSQUIC.so')


# Scalar SQUIC Paramter Runtime
Y_train = Y1
Y_test  = None


W0 = None
X0 = None

M = None
l=.25

max_itr=100
term_tol=1e-3
drop_tol=1e-3
verbose=1

[X,W,info_obj_trace,info_time_list,info_logdetX_Y1,info_trXS_Y2,info_dgap]=SQUIC.run(Y_train,Y_test,l,M,X0,W0,max_itr,term_tol,drop_tol,verbose)
print("Inverse Covariance Matrix \n", X.todense())
print("Covariance Matrix \n", W.todense())
print("Objective Function Trace\n", info_obj_trace)
print("Time List [time_total,time_impcov,time_optimz,time_factor,time_aprinv,time_updte] \n", info_time_list)
print("info_logdetX_Y1 \n", info_logdetX_Y1)
print("info_trXS_Y2 \n", info_trXS_Y2)
print("info_dgap \n", info_dgap)



# Matrix SQUIC Paramter Runtime
Y_train = Y1
Y_test  = None


W0 = None
X0 = None

# The M matrix forms the matrix penelty matrix. Let Lambda be matrix penelty parameter and l be scalar penelty paramter:
# Than Lambda :=  M + l(1-pattern(M)). 
# This mean M encodes as bias, the structure of the inverse covariance matrix with nonzero values equaling the 
M      = csr_matrix((p,p))

# For example make tridiagioanl structure
for i in range(0,p):

	j=i
	M[i,j]=1;

	j=i+1
	if(j<p):
		M[i,j]=1;
		M[j,i]=1;

M = M* 1e-6;


l=.5

max_itr=100
term_tol=1e-3
drop_tol=1e-3
verbose=1

[X,W,info_obj_trace,info_time_list,info_logdetX_Y1,info_trXS_Y2,info_dgap]=SQUIC.run(Y_train,Y_test,l,M,X0,W0,max_itr,term_tol,drop_tol,verbose)
print("Inverse Covariance Matrix \n", X.todense())
print("Covariance Matrix \n", W.todense())
print("Objective Function Trace\n", info_obj_trace)
print("Time List [time_total,time_impcov,time_optimz,time_factor,time_aprinv,time_updte] \n", info_time_list)
print("info_logdetX_Y1 \n", info_logdetX_Y1)
print("info_trXS_Y2 \n", info_trXS_Y2)
print("info_dgap \n", info_dgap)







