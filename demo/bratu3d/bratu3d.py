# ------------------------------------------------------------------------
#
#  Solid Fuel Ignition (SFI) problem.  This problem is modeled by the
#  partial differential equation
#
#          -Laplacian(u) - lambda * exp(u) = 0,  0 < x,y,z < 1,
#
#  with boundary conditions
#
#           u = 0  for  x = 0, x = 1, y = 0, y = 1, z = 0, z = 1
#
#  A finite difference approximation with the usual 7-point stencil
#  is used to discretize the boundary value problem to obtain a
#  nonlinear system of equations. The problem is solved in a 3D
#  rectangular domain, using distributed arrays (DAs) to partition
#  the parallel grid.
#
# ------------------------------------------------------------------------

import sys, petsc4py
petsc4py.init(sys.argv)

from numpy import exp, sqrt
from petsc4py import PETSc

class Bratu3D(object):

    def __init__(self, da, lambda_):
        assert da.getDim() == 3
        self.da = da
        self.lambda_ = lambda_
        self.localX  = da.createLocalVector()

    def formInitGuess(self, snes, X):
        #
        X.zeroEntries()
        corners, sizes = self.da.getCorners()
        x = X[...].reshape(sizes, order='f')
        #
        mx, my, mz = self.da.getSizes()
        hx, hy, hz = [1.0/m for m in [mx, my, mz]]
        lambda_ = self.lambda_
        scale = lambda_/(lambda_ + 1.0)
        #
        (xs, xe), (ys, ye), (zs, ze) = self.da.getRanges()
        for k in xrange(zs, ze):
            min_k = min(k,mz-k-1)*hz
            for j in xrange(ys, ye):
                min_j = min(j,my-j-1)*hy
                for i in xrange(xs, xe):
                    min_i = min(i,mx-i-1)*hx
                    if (i==0    or j==0    or k==0 or
                        i==mx-1 or j==my-1 or k==mz-1):
                        # boundary points
                        x[i, j, k] = 0.0
                    else:
                        # interior points
                        min_kij = min(min_i,min_j,min_k)
                        x[i, j, k] = scale*sqrt(min_kij)

    def formFunction(self, snes, X, F):
        #
        self.da.globalToLocal(X, self.localX)
        corners, sizes = self.da.getGhostCorners()
        x = self.localX[...].reshape(sizes, order='f')
        #
        F.zeroEntries()
        corners, sizes = self.da.getCorners()
        f = F[...].reshape(sizes, order='f')
        #
        mx, my, mz = self.da.getSizes()
        hx, hy, hz = [1.0/m for m in [mx, my, mz]]
        hxhyhz  = hx*hy*hz
        hxhzdhy = hx*hz/hy;
        hyhzdhx = hy*hz/hx;
        hxhydhz = hx*hy/hz;
        lambda_ = self.lambda_
        #
        (xs, xe), (ys, ye), (zs, ze) = self.da.getRanges()
        for k in xrange(zs, ze):
            for j in xrange(ys, ye):
                for i in xrange(xs, xe):
                    if (i==0    or j==0    or k==0 or
                        i==mx-1 or j==my-1 or k==mz-1):
                        f[i, j, k] = x[i, j, k] - 0
                    else:
                        u   = x[ i  ,  j  ,  k ] # center
                        u_e = x[i+1 ,  j  ,  k ] # east
                        u_w = x[i-1 ,  j  ,  k ] # west
                        u_n = x[ i  , j+1 ,  k ] # north
                        u_s = x[ i  , j-1 ,  k ] # south
                        u_u = x[ i  ,  j  , k+1] # up
                        u_d = x[ i  ,  j  , k-1] # down
                        u_xx = (-u_e + 2*u - u_w)*hyhzdhx
                        u_yy = (-u_n + 2*u - u_s)*hxhzdhy
                        u_zz = (-u_u + 2*u - u_d)*hxhydhz
                        f[i, j, k] = u_xx + u_yy + u_zz \
                                     - lambda_*exp(u)*hxhyhz

    def formJacobian(self, snes, X, J, P):
        raise NotImplementedError
        P.zeroEntries()
        if J != P: J.assemble() # matrix-free operator
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN


OptDB = PETSc.Options()

N = OptDB.getInt('N', 16)
lambda_ = OptDB.getReal('lambda', 6.0)
do_plot = OptDB.getBool('plot', False)

da = PETSc.DA().create([N, N, N])
pde = Bratu3D(da, lambda_)

snes = PETSc.SNES().create()
F = da.createGlobalVector()
snes.setFunction(pde.formFunction, F)

fd = OptDB.getBool('fd', True)
mf = OptDB.getBool('mf', False)
if mf:
    J = None
    snes.setUseMF()
else:
    J = da.createMatrix()
    snes.setJacobian(pde.formJacobian, J)
    if fd:
        snes.setUseFD()

X = da.createGlobalVector()
pde.formInitGuess(None, X)

snes.getKSP().setType('cg')
snes.setFromOptions()
snes.solve(None, X)

U = da.createNaturalVector()
da.globalToNatural(X, U)

def plot(da, U):
    comm = da.getComm()
    scatter, U0 = PETSc.Scatter.toZero(U)
    scatter.scatter(U, U0, False, PETSc.Scatter.Mode.FORWARD)
    rank = comm.getRank()
    if rank == 0:
        solution = U0[...]
        solution = solution.reshape(da.sizes, order='f').copy()
        try:
            from matplotlib import pyplot
            pyplot.contourf(solution[:, :, N//2])
            pyplot.axis('equal')
            pyplot.show()
        except:
            raise
            pass
    comm.barrier()
    scatter.destroy()
    U0.destroy()

if do_plot: plot(da, U)

del pde, da, snes
del F, J, X, U
