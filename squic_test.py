import SQUIC_Py as SQUIC_Py
import numpy as np
from scipy.sparse import csr_matrix,isspmatrix_csr, identity


def tridiag(off_diag, diag, p):
	a=np.ones(p-1)*off_diag
	b=np.ones(p)*diag
	return np.diag(a,-1) + np.diag(b,0) + np.diag(a,1)


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


p=20
n=10



np.random.seed(1)
iC_star = tridiag(-0.5,1.25,p)
L = np.linalg.cholesky(iC_star)
Y = np.linalg.solve(L.T,np.random.randn(p,n))

S=np.cov(Y,bias=1)


print("There is n=%d samples and p=%d variables"% (n,p));
print("Sample Covariance Matrix (dont need this just for info): \n",S);
if(n>p):
	print("Inverse of Sample Covariance Matrix (dont need this just for info): \n",np.linalg.inv(S));
print("True Inverse Covariance Matrix: \n",iC_star);

# Set the SQUIC Library Path
SQUIC_Py.set_path('/Users/aryan/gdrive/files/code/SQUIC_Release_Source/darwin20/libSQUIC.dylib')
#SQUIC.set_path('/local_home/aryan/SQUIC_Release_Source/linux/libSQUIC.so')



# Scalar SQUIC Paramter Runtime
l=.25

[X,W,info_times,info_objective,info_logdetX,info_trSX]=SQUIC_Py.SQUIC(Y=Y,l=l)
print("Inverse Covariance Matrix \n", X.todense())
print("Covariance Matrix \n", W.todense())
print("Objective Function Trace\n", info_objective)
print("Time List [time_total,time_impcov,time_optimz,time_factor,time_aprinv,time_updte] \n", info_times)
print("info_logdetX \n", info_logdetX)
print("info_trSX \n", info_trSX)



# Matrix SQUIC Paramter Runtime
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


[X,W,info_times,info_objective,info_logdetX,info_trSX]=SQUIC_Py.SQUIC(Y=Y,l=l,M=M)
print("Inverse Covariance Matrix \n", X.todense())
print("Covariance Matrix \n", W.todense())
print("Objective Function Trace\n", info_objective)
print("Time List [time_total,time_impcov,time_optimz,time_factor,time_aprinv,time_updte] \n", info_times)
print("info_logdetX \n", info_logdetX)
print("info_trSX \n", info_trSX)


[S,info_times]=SQUIC_Py.SQUIC_S(Y=Y,l=l)
print("Sample Covariance Matrix \n", S.todense())
print("Time List [time_total,time_impcov,time_optimz,time_factor,time_aprinv,time_updte] \n", info_times)








