try:
    execfile
except NameError:
    def execfile(file, globals=globals(), locals=locals()):
        fh = open(file, "r")
        try: exec(fh.read()+"\n", globals, locals)
        finally: fh.close()

import petsc4py, sys
petsc4py.init(sys.argv)

from petsc4py import PETSc

execfile('petsc-mat.py')
execfile('petsc-cg.py')

x, b = A.createVecs()

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
print("iterations: %d residual norm: %g" % (ksp.its, ksp.norm)) 

x.set(0)
b.set(1)
its, norm = cg(A,b,x,100,1e-5)
print("iterations: %d residual norm: %g" % (its, norm)) 

OptDB = PETSc.Options()

if OptDB.getBool('plot', True):
    da = PETSc.DMDA().create([m,n])
    u = da.createGlobalVec()
    x.copy(u)
    draw = PETSc.Viewer.DRAW()
    OptDB['draw_pause'] = 1
    draw(u)

if OptDB.getBool('plot_mpl', False):
    try:
        from matplotlib import pylab
    except ImportError:
        print("matplotlib not available")
    else:
        from numpy import mgrid
        X, Y =  mgrid[0:1:1j*m,0:1:1j*n]
        Z = x[...].reshape(m,n)
        pylab.figure()
        pylab.contourf(X,Y,Z)
        pylab.plot(X.ravel(),Y.ravel(),'.k')
        pylab.axis('equal')
        pylab.colorbar()
        pylab.show()
