import numpy as np
import scipy as sc


#determine which velocities and diretions to use based on position within a grid

def active_grid(x,y):
    return int(np.trunc(x)), int(np.trunc(y))
    

def kernel_func(x,y,sigma,L):
    return sigma**2*np.exp(-((x-y)/L)**2.0)  #72 is the extrapolation as each time step is now 3 days/72 hours    


#kernel function will take in a numpy array and output a numpy covariance matrix
#the numpy array's first column should be the distance label to use in the kernel
#kernel_1 computes the square covariance matrices
#kernel_2 computes the non-square covariances
def kernel_1(x,sigma,L):
    size = x.shape[0]
    cov = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            cov[i,j] = kernel_func(x[i,0],x[j,0],sigma,L)  
    return cov

#accepts two arrays and computes the cross kernels
def kernel_2(x1,x2,sigma,L):
    #x1 should be the conditioned vector, x2 the conditioning vector
    size_x1 = x1.shape[0]
    size_x2 = x2.shape[0]
    cov = np.zeros((size_x1,size_x2))
    for i in range(size_x1):
        for j in range(size_x2):
            cov[i,j] = kernel_func(x1[i,0],x2[j,0],sigma,L)
    return cov



def cond_mean(m1,m2,cov12,cov22,r,x2):
    m = cov22 + r*np.identity(cov22.shape[0])
    m_inv = sc.linalg.inv(m)
    prod1 = np.matmul(cov12,m_inv)
    prod2 = np.matmul(prod1,(x2 - m2).reshape((-1,1)))
    cond_mean = m1 + prod2
    return cond_mean

def cond_var(cov11,cov12,cov22,r):
    m = cov22 + r*np.identity(cov22.shape[0])
    m_inv = sc.linalg.inv(m)
    prod1 = np.matmul(cov12,m_inv)
    prod2 = np.matmul(prod1,np.transpose(cov12))
    cond_var = cov11 - prod2
    return cond_var





            
            
    





