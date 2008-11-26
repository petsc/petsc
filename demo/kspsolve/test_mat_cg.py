import petsc4py, sys
petsc4py.init(sys.argv)

from petsc4py import PETSc

execfile('petsc-mat.py')
execfile('petsc-cg.py')

x, b = A.getVecs()

ksp = PETSc.KSP().create()
ksp.setType('cg')
ksp.getPC().setType('none')
ksp.setOperators(A)
ksp.setFromOptions()

ksp.max_it = 100
ksp.rtol = 1e-5
ksp.atol = 0
x.set(0)
b.set(1)
ksp.solve(b,x)
print ksp.its, ksp.rnorm

x.set(0)
b.set(1)
its, rnorm = cg(A,b,x,100,1e-5)
print its, rnorm

try:
    from matplotlib import pylab
except ImportError:
    raise SystemExit("matplotlib not available")
from numpy import mgrid
X, Y =  mgrid[0:1:1j*m,0:1:1j*n]
Z = x[...].reshape(m,n)
pylab.figure()
pylab.contourf(X,Y,Z)
pylab.plot(X.ravel(),Y.ravel(),'.k')
pylab.axis('equal')
pylab.colorbar()
pylab.show()
