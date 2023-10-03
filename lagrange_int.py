
def Lagrangebasis(x_int, x):
    """Construct Lagrange basis function for points in xj
    
    Parameters
    ----------
    x_int : array
        Interpolation points
    x : Sympy Symbol
    
    Returns
    -------
    ell : list 
          element j is j-th basis function for j=0,...,k.  
    Lagrange basis functions 0,...,k for interpolating points xj = (x0, ..., xk) 
    """
    from sympy import Mul
    n = len(x_int) # n = k+1
    ell = []
    numerator_full = Mul(*[x - x_int[m] for m in range(n)]) 

    for j in range(n):
        numerator = numerator_full / (x - x_int[j])
        denominator = Mul(*[x_int[j] - x_int[m] for m in range(n) if j!=m])    
        ell.append(numerator/denominator)
    return ell 


def Lagrangefunction(u, basis):
    """Return Lagrange polynomial
    
    Parameters
    ----------
    u : array
        Mesh function values
    basis : tuple of Lagrange basis functions
        Output from Lagrangebasis
    """
    f = 0
    for j, uj in enumerate(u):
        f += basis[j]*uj
    return f


def Lagrangefunction2D(u, basisx, basisy):
    """
    using equal number of points (so equal no. of basis functions) in each coordinate direction
    len(basisx), len(basisy) == u.shape must be satisfies 

    Args:
        u (array): 2D mesh function
        basisx (list): list of lagrange basis functions in x (output of lagrangebasis)
        basisy (list): list of lagrange basis functions in y (output of lagrangebasis)
    """
    # assert (u.shape[0] == len(basisx) and u.shape[1] == len(basisy))
    f = 0 
    m = len(basisx)
    n = len(basisy)
    #m, n = u.shape
    for i in range(m):
        for j in range(n):
            f += u[i, j] * basisx[i] * basisy[j]
    return f 

if __name__=='__main__':
    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt 

    x, y = sp.symbols('x y')
    def mesh2D(Nx, Ny, Lx, Ly, sparse=False):
        x = np.linspace(0, Lx, Nx+1)
        y = np.linspace(0, Ly, Ny+1)
        return np.meshgrid(x, y, indexing='ij', sparse=sparse)

    N = 10
    xij, yij = mesh2D(N, N, 1, 1, False)
    u2 = xij*(1-xij)*yij*(1-yij)
    plt.figure(figsize=(3, 2))
    plt.contourf(xij, yij, u2);

    y = sp.Symbol('y')
    lx = Lagrangebasis(xij[5:7, 0], x=x)
    ly = Lagrangebasis(yij[0, 6:8], x=y)

    f = Lagrangefunction2D(u2[5:7, 6:8], lx, ly)
    print(sp.simplify(f))
    ue = x*(1-x)*y*(1-y)
    print(f.subs({x: 0.55, y: 0.65}), ue.subs({x: 0.55, y: 0.65}))

    xp, yp = 0.55, 0.65
    k = 2 # order of approximation 
    x_values = xij[:, 0]
    y_values = yij[0, :]
    # get indices of k closest mesh points in each direction
    x_inds = np.argsort(np.abs(x_values - xp))[:k+1]
    y_inds = np.argsort(np.abs(y_values - yp))[:k+1] 

    x_int = np.sort(x_values[x_inds]) #[::-1]
    y_int = np.sort(y_values[y_inds]) #[::-1]

    x_basis = Lagrangebasis(x_int, sp.symbols('x'))
    y_basis = Lagrangebasis(y_int, sp.symbols('y'))

    print(x_int)
    print(y_int)

    f = Lagrangefunction2D(u2, x_basis, y_basis)
    print(f.subs({x: 0.55, y: 0.65}), ue.subs({x: 0.55, y: 0.65}))

