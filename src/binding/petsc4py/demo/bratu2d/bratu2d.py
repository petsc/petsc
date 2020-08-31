import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

# this user class is an application
# context for the nonlinear problem
# at hand; it contains some parametes
# and knows how to compute residuals

class Bratu2D:

    def __init__(self, nx, ny, alpha, impl='python'):
        self.nx = nx # x grid size
        self.ny = ny # y grid size
        self.alpha = alpha
        if impl == 'python':
            from bratu2dnpy import bratu2d
            order = 'c'
        elif impl == 'fortran':
            from bratu2df90 import bratu2d
            order = 'f'
        else:
            raise ValueError('invalid implementation')
        self.compute = bratu2d
        self.order = order

    def evalFunction(self, snes, X, F):
        nx, ny = self.nx, self.ny
        alpha = self.alpha
        order = self.order
        x = X.getArray(readonly=1).reshape(nx, ny, order=order)
        f = F.getArray(readonly=0).reshape(nx, ny, order=order)
        self.compute(alpha, x, f)

# convenience access to
# PETSc options database
OptDB = PETSc.Options()

nx = OptDB.getInt('nx', 32)
ny = OptDB.getInt('ny', nx)
alpha = OptDB.getReal('alpha', 6.8)
impl  = OptDB.getString('impl', 'python')

# create application context
# and PETSc nonlinear solver
appc = Bratu2D(nx, ny, alpha, impl)
snes = PETSc.SNES().create()

# register the function in charge of
# computing the nonlinear residual
f = PETSc.Vec().createSeq(nx*ny)
snes.setFunction(appc.evalFunction, f)

# configure the nonlinear solver
# to use a matrix-free Jacobian
snes.setUseMF(True)
snes.getKSP().setType('cg')
snes.setFromOptions()

# solve the nonlinear problem
b, x = None, f.duplicate()
x.set(0) # zero inital guess
snes.solve(b, x)

if OptDB.getBool('plot', True):
    da = PETSc.DMDA().create([nx,ny])
    u = da.createGlobalVec()
    x.copy(u)
    draw = PETSc.Viewer.DRAW()
    OptDB['draw_pause'] = 1
    draw(u)

if OptDB.getBool('plot_mpl', False):
    try:
        from matplotlib import pylab
    except ImportError:
        PETSc.Sys.Print("matplotlib not available")
    else:
        from numpy import mgrid
        X, Y =  mgrid[0:1:1j*nx,0:1:1j*ny]
        Z = x[...].reshape(nx,ny)
        pylab.figure()
        pylab.contourf(X,Y,Z)
        pylab.colorbar()
        pylab.plot(X.ravel(),Y.ravel(),'.k')
        pylab.axis('equal')
        pylab.show()
