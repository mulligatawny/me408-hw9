import numpy as np

#=============================================================================#
# Discrete 1D Cosine Transform (Forward and Reverse) from Moin P. pp. 175     #
#=============================================================================#

def DCT(xj, f):
    """
    Computes the 1D discrete cosine transform of f
    Parameters:
        xj (numpy array) : grid points
        f  (lambda func) : function
    Returns:
        Fk (numpy array) : DCT coefficients

    """
    N = int(len(xj))-1
    Fk = np.zeros_like(xj, dtype='float')

    for k in range(N+1):
        ak = 0.0
        for j in range(N+1):
            if j == 0 or j == N:
                ak = ak + f(xj[j])/2.0*np.cos(k*xj[j])
            else:
                ak = ak + f(xj[j])*np.cos(k*xj[j])
        if k == 0 or k == N:
            ak = ak/N
        else:
            ak = ak*2/N
        Fk[k] = ak
    return Fk

def IDCT(xj, Fk):
    """
    Computes the 1D discrete inverse cosine transform of f
    Parameters:
        xj (numpy array) : grid points
        Fk (numpy array) : DCT coefficients
    Returns:
        fc (numpy array) : reconstructed function 

    """
    fc = np.zeros_like(xj, dtype='float')
    N = int(len(xj))-1

    for k in range(N+1):
        fc = fc + Fk[k]*np.cos(k*xj)
    return fc
