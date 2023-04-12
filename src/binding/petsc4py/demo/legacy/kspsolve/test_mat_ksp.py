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
execfile('petsc-ksp.py')

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
