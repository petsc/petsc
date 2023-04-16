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

try: range = xrange
except: pass

import sys, petsc4py
petsc4py.init(sys.argv)

from numpy import exp, sqrt
from petsc4py import PETSc

class Bratu3D(object):

    def __init__(self, da, lambda_):
        assert da.getDim() == 3
        self.da = da
        self.lambda_ = lambda_
        self.localX  = da.createLocalVec()

    def formInitGuess(self, snes, X):
        #
        x = self.da.getVecArray(X)
        #
        mx, my, mz = self.da.getSizes()
        hx, hy, hz = [1.0/(m-1) for m in [mx, my, mz]]
        lambda_ = self.lambda_
        scale = lambda_/(lambda_ + 1.0)
        #
        (xs, xe), (ys, ye), (zs, ze) = self.da.getRanges()
        for k in range(zs, ze):
            min_k = min(k,mz-k-1)*hz
            for j in range(ys, ye):
                min_j = min(j,my-j-1)*hy
                for i in range(xs, xe):
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
        x = self.da.getVecArray(self.localX)
        f = self.da.getVecArray(F)
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
        for k in range(zs, ze):
            for j in range(ys, ye):
                for i in range(xs, xe):
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
        #
        self.da.globalToLocal(X, self.localX)
        x = self.da.getVecArray(self.localX)
        #
        mx, my, mz = self.da.getSizes()
        hx, hy, hz = [1.0/m for m in [mx, my, mz]]
        hxhyhz  = hx*hy*hz
        hxhzdhy = hx*hz/hy;
        hyhzdhx = hy*hz/hx;
        hxhydhz = hx*hy/hz;
        lambda_ = self.lambda_
        #
        P.zeroEntries()
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()
        #
        (xs, xe), (ys, ye), (zs, ze) = self.da.getRanges()
        for k in range(zs, ze):
            for j in range(ys, ye):
                for i in range(xs, xe):
                    row.index = (i,j,k)
                    row.field = 0
                    if (i==0    or j==0    or k==0 or
                        i==mx-1 or j==my-1 or k==mz-1):
                        P.setValueStencil(row, row, 1.0)
                    else:
                        u = x[i,j,k]
                        diag = (2*(hyhzdhx+hxhzdhy+hxhydhz)
                                - lambda_*exp(u)*hxhyhz)
                        for index, value in [
                            ((i,j,k-1), -hxhydhz),
                            ((i,j-1,k), -hxhzdhy),
                            ((i-1,j,k), -hyhzdhx),
                            ((i, j, k), diag),
                            ((i+1,j,k), -hyhzdhx),
                            ((i,j+1,k), -hxhzdhy),
                            ((i,j,k+1), -hxhydhz),
                            ]:
                            col.index = index
                            col.field = 0
                            P.setValueStencil(row, col, value)
        P.assemble()
        if J != P: J.assemble() # matrix-free operator
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

OptDB = PETSc.Options()

n  = OptDB.getInt('n', 16)
nx = OptDB.getInt('nx', n)
ny = OptDB.getInt('ny', n)
nz = OptDB.getInt('nz', n)
lambda_ = OptDB.getReal('lambda', 6.0)

da = PETSc.DMDA().create([nx, ny, nz], stencil_width=1)
pde = Bratu3D(da, lambda_)

snes = PETSc.SNES().create()
F = da.createGlobalVec()
snes.setFunction(pde.formFunction, F)

fd = OptDB.getBool('fd', False)
mf = OptDB.getBool('mf', False)
if mf:
    J = None
    snes.setUseMF()
else:
    J = da.createMat()
    snes.setJacobian(pde.formJacobian, J)
    if fd:
        snes.setUseFD()

snes.getKSP().setType('cg')
snes.setFromOptions()

X = da.createGlobalVec()
pde.formInitGuess(snes, X)
snes.solve(None, X)

U = da.createNaturalVec()
da.globalToNatural(X, U)

if OptDB.getBool('plot_mpl', False):

    def plot_mpl(da, U):
        comm = da.getComm()
        rank = comm.getRank()
        scatter, U0 = PETSc.Scatter.toZero(U)
        scatter.scatter(U, U0, False, PETSc.Scatter.Mode.FORWARD)
        if rank == 0:
            try:
                from matplotlib import pylab
            except ImportError:
                PETSc.Sys.Print("matplotlib not available")
            else:
                from numpy import mgrid
                nx, ny, nz = da.sizes
                solution = U0[...].reshape(da.sizes, order='f')
                xx, yy =  mgrid[0:1:1j*nx,0:1:1j*ny]
                pylab.contourf(xx, yy, solution[:, :, nz//2])
                pylab.axis('equal')
                pylab.xlabel('X')
                pylab.ylabel('Y')
                pylab.title('Z/2')
                pylab.show()
        comm.barrier()

    plot_mpl(da, U)
