###############################################################################
# 1D Chebyshev Routines:                                                      #
# Discrete Chebyshev Transforms (Forward and Reverse) from Moin P. pp. 190    #
# Derivative using Chebyshev Transforms                                       #
###############################################################################

import numpy as np

def cheby(f):
    """
    Computes the 1D discrete Chebyshev transform of f
    Parameters:
        f  (array_like) : function
    Returns:
        Fk (array)      : Chebyshev coefficients

    """
    N = int(len(f))-1
    Fk = np.zeros_like(f, dtype='float')
    t = np.arange(0, N+1)*np.pi/N # uniform grid in theta
    
    for k in range(N+1):
        cs = np.cos(k*t)
        cs[0] = cs[0]/2
        cs[-1] = cs[-1]/2
        Fk[k] = np.dot(f,cs)/N*2
    Fk[0] = Fk[0]/2
    Fk[-1] = Fk[-1]/2
    return Fk

def icheby(Fk):
    """
    Computes the 1D discrete inverse Chebyshev transform of f
    Parameters:
        Fk (array_like) : Chebyshev coefficients
    Returns:
        fc (array)      : reconstructed function 

    """
    N = int(len(Fk))-1
    fc = np.zeros_like(Fk, dtype='float')
    t = np.arange(0, N+1)*np.pi/N # uniform grid in theta

    for k in range(N+1):
        fc = fc + Fk[k]*np.cos(k*t)
    return fc

def cheby_der(f):
    """
    Computes the derivative of a function using Chebyshev transforms
    Parameters:
        f  (array_like) : function
    Returns:
        phi (array)      : chebyshev coefficients of derivative of f

    """
    N = len(f)-1
    # compute chebyshev transform
    Fk = cheby(f)
    k = np.arange(0, N+1)
    # assemble bi-diagonal matrix
    A = np.zeros((N+1, N+1))
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:,1:], -1)
    A[0,:] = 0
    A[1,0] = 2
    nA = A[1:,:-1]
    # assmble RHS
    b = np.zeros(N+1)
    b = 2*k*Fk
    bn = b[1:]
    # solve bi-diagonal system
    phi = np.linalg.solve(nA, bn)
    # set last coefficient to 0
    phi = np.append(phi, 0.0)
    # inverse transform
    fp = icheby(phi)
    return fp
