import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt 
from lagrange_int import Lagrangebasis, Lagrangefunction2D

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N 
        self.h = self.L / N 
        x = np.linspace(0, self.L, self.N+1)
        y = np.linspace(0, self.L, self.N+1)
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij')


    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        D /= self.h**2
        return D

    def laplace(self):
        """Return a vectorized Laplace operator, valid at interior points"""
        D2 = self.D2()
        A = sparse.kron(D2, sparse.eye(self.N+1)) + sparse.kron(sparse.eye(self.N + 1), D2)
        return A 

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((self.N+1, self.N+1), dtype=bool)
        B[1:-1, 1:-1] = 0 
        binds = np.where(B.ravel() == 1)[0]
        return binds 

    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        
        A = self.laplace() 
        binds = self.get_boundary_indices()
        A = A.tolil()
        for i in binds:
            A[i] = 0 
            A[i, i] = 1 
        A = A.tocsr()    

        # self.x/y must be 2D (i.e. not from sparse) to make sure F is correct shape
        ONE = np.ones((self.N+1, self.N+1))
        F = sp.lambdify((x,y), self.f)(self.xij, self.yij) # * ONE # hack in case f is constant
        b = F.ravel()
        b[binds] = (sp.lambdify((x,y), self.ue)(self.xij, self.yij) * ONE).ravel()[binds]  
        return A, b 

    def l2_error(self, u):
        u_exact = sp.lambdify((x,y), self.ue)(self.xij, self.yij)
        return np.sqrt(self.h**2*np.sum((u_exact-u)**2)), u_exact # multiply with h**2 or use np.mean instead of np.sum 

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((N+1, N+1))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            u = self(N0)
            E.append(self.l2_error(u)[0])
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, xp, yp, k=2):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation (not necessarily a mesh pt.)
        k    : int
                order of approximation
        Returns
        -------
        The value of u(x, y)

        """
        
        x_values = self.xij[:, 0] # index processing
        y_values = self.yij[0, :]
        x_inds = np.argsort(np.abs(x_values - xp))[:k+1]
        y_inds = np.argsort(np.abs(y_values - yp))[:k+1] 
        x_inds = np.sort(x_inds) # make sure indices are increasing
        y_inds = np.sort(y_inds)
        
        x_int = x_values[x_inds] # interpolation points
        y_int = y_values[y_inds] 

        x_basis = Lagrangebasis(x_int, x)
        y_basis = Lagrangebasis(y_int, y)

        u = self.U[x_inds][:, y_inds]
        f = Lagrangefunction2D(u, x_basis, y_basis)
        return f.subs({x:xp, y:yp})


def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    interpolate_at_point(sol, 0.52, 0.63, 1)
    interpolate_at_point(sol, sol.h/2, 1-sol.h/2)
    #assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    #assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h, y: 1-sol.h/2}).n()) < 1e-3

def interpolate_at_point(sol, a, b, k=2):
    diff = abs(sol.eval(a, b, k) - ue.subs({x: a, y: b}).n()) 
    print(f'Interpolation error: {diff}, k={k} at {(a,b)}')
    assert diff < 1e-3


if __name__=='__main__':
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)

    u_solved = sol(N=100)
    err, u_exact = sol.l2_error(u_solved)
    test_interpolation()

