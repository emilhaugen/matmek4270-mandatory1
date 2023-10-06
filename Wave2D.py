import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:
    """Class for solving the wave equation in unit square with homogeneous Dirichlet boundary.

    Parameters
    ----------
    N : int
        Number of uniform spatial intervals

    c0 : number, optional
        The wavespeed
    cfl : number, optional
        CFL number
    """

    def __init__(self, cfl=0.5, c=1.0, mx=3, my=3):
        self.c = c
        self.cfl = cfl
        self.mx = mx 
        self.my = my 


    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        self.h = 1 / N
        x = np.linspace(0, 1, N+1)
        y = np.linspace(0, 1, N+1)
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij', sparse=sparse)


    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        D /= self.h**2
        return D


    @property
    def w(self):
        """Return the dispersion coefficient. Computed by inserting excact solution into the equation."""
        kx = np.pi * self.mx 
        ky = np.pi * self.my
        w = np.sqrt(kx**2 + ky**2) * self.c  
        return w  

    def ue(self):
        """Return the exact standing wave"""
        return sp.sin(self.mx*sp.pi*x)*sp.sin(self.my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$ 
                                            --> n=0, n=1???

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        D : array 
            differentiation matrix 
        e : tuple 
            l2 error at first two time steps        
        """
        # follwing array is automatically zero at spatial boundaries
        self.Unm1 = U0 = sp.lambdify((x,y,t), self.ue())(self.xij, self.yij, 0) # t = 0 
        D = self.D2()
        # time derivative at t=0 is zero and given as initial data
        # so we get the following formula for U1:
        U1 = U0 + 0.5 * (self.c * self.dt)**2  * (D @ U0 + U0 @ D.T)
        self.Un = U1 
        self.Unp1 = np.zeros_like(U1)
        self.apply_bcs()
        e0 = self.l2_error(self.Unm1, 0)
        e1 = self.l2_error(self.Un, self.dt)
        return D, (e0, e1)


    @property
    def dt(self):
        """Return the time step"""
        return self.cfl * self.h / self.c 

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        ue = sp.lambdify((x,y,t), self.ue())(self.xij, self.yij, t0)
        return np.sum(self.h**2 * (u - ue).flatten()**2)**(.5)

    def apply_bcs(self):
        """Slightly inefficeint but cleaner. Since sin(k*pi)=0 for any integer k, we 
        always get zero along the spatial boundaries."""
        for U in (self.Unm1, self.Un, self.Unp1):
            U[0] = 0 
            U[-1] = 0 
            U[:, 0] = 0 
            U[:, -1] = 0 


    def __call__(self, N, Nt, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """

        self.create_mesh(N)

        l2_error = np.zeros(Nt+1)
        D, initial_errors = self.initialize()
        l2_error[0:2] = initial_errors

        plot_data = {0 : self.Unm1.copy()}

        if store_data == 1:
            plot_data[1] = self.Un.copy() 

        print(f'Solving for t in {[0, self.dt*Nt]} with (N, Nt) = {N, Nt}.')

        for n in range(1, Nt):
            self.Unp1 = 2 * self.Un - self.Unm1 + \
                (self.c * self.dt)**2 * (D @ self.Un + self.Un @ D.T)   
            self.apply_bcs()
            self.Unm1 = self.Un 
            self.Un = self.Unp1
            l2_error[n+1] = self.l2_error(self.Un, (n+1)*self.dt)
            if n % store_data == 0:
                plot_data[n] = self.Unm1.copy()  
        
        if store_data == -1:
            return self.h, l2_error
        
        return self.xij, self.yij, plot_data

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors (at time T)
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):


    def __init__(self, cfl=0.5, c=1.0, mx=3, my=3):
        super().__init__(cfl, c, mx, my)


    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D[0, :2] = -2, 2
        D[-1, -2:] = 2, -2
        D /= self.h**2
        return D


    def ue(self):
        """Return the exact standing wave"""
        return sp.cos(self.mx*sp.pi*x)*sp.cos(self.my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        pass 

def test_convergence_wave2d():
    sol = Wave2D(cfl=0.1, c=1, mx=2, my=3)
    r, E, h = sol.convergence_rates(m=4)
    print(r)
    print(E)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann(cfl=0.1, c=2, mx=2, my=3)
    r, E, h = solN.convergence_rates(m=4)
    print(r)
    print(E)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    m = 2 
    C = 2**(-.5)

    sol = Wave2D(cfl=C, c=1, mx=m, my=m)
    r, E, h = sol.convergence_rates(m=4)
    assert E.max() < 1e-14

    sol = Wave2D_Neumann(cfl=C, c=1, mx=m, my=m)
    r, E, h = sol.convergence_rates(m=4)
    assert E.max() < 1e-14


if __name__=='__main__':

    test_exact_wave2d()
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()


