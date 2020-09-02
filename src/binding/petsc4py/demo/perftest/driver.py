#!/usr/bin/env python
import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

try:
    from matplotlib import pylab
except ImportError:
    pylab = None

# this user class is an application
# context for the nonlinear problem
# at hand; it contains some parametes
# and knows how to compute residuals

class AppCtx:

    def __init__(self, nx, ny, nz):
        self.n = np.array([nx, ny, nz], dtype='i')
        self.h = np.array([1.0/(n-1) for n in self.n], dtype='d')
        from App import formFunction
        from App import formInitial
        self._formFunction = formFunction
        self._formInitial = formInitial

    def formInitial(self, t, X):
        xx = X.getArray(readonly=0).reshape(self.n, order='f')
        self._formInitial(self.h, t, xx)

    def formFunction(self, ts, t, X, Xdot, F):
        n = self.n
        h = self.h
        x = X.getArray(readonly=1).reshape(n, order='f')
        xdot = Xdot.getArray(readonly=1).reshape(n, order='f')
        f = F[...].reshape(n, order='f')
        self._formFunction(h, t, x, xdot, f)

    def plot(self, t, x):
        nx, ny, nz = self.n
        from numpy import mgrid
        #
        U = x.getArray(readonly=1).reshape(nx,ny,nz, order='f')
        #
        X, Y =  mgrid[0:1:1j*nx,0:1:1j*ny]
        Z = U[:,:,nz//2]
        pylab.figure(0)
        pylab.contourf(X,Y,Z)
        pylab.colorbar()
        pylab.plot(X.ravel(),Y.ravel(),'.k')
        pylab.title('z=0.50')
        pylab.xlabel('x')
        pylab.ylabel('y')
        pylab.axis('equal')
        #
        X, Y =  mgrid[0:1:1j*nx,0:1:1j*nz]
        Z = U[:,ny//4,:]
        pylab.figure(1)
        pylab.contourf(X,Y,Z)
        pylab.colorbar()
        pylab.plot(X.ravel(),Y.ravel(),'.k')
        pylab.title('y=0.25')
        pylab.xlabel('x')
        pylab.ylabel('z')
        pylab.axis('equal')
        #
        X, Y =  mgrid[0:1:1j*ny,0:1:1j*nz]
        Z = U[nx//2,:,:]
        pylab.figure(2)
        pylab.contourf(X,Y,Z)
        pylab.colorbar()
        pylab.plot(X.ravel(),Y.ravel(),'.k')
        pylab.title('x=0.50')
        pylab.xlabel('y')
        pylab.ylabel('z')
        pylab.axis('equal')


def run_test(nx,ny,nz,samples,plot=False):
    ts  = PETSc.TS().create()
    ts.setType('theta')
    ts.setTheta(1.0)
    ts.setTimeStep(0.01)
    ts.setTime(0.0)
    ts.setMaxTime(1.0)
    ts.setMaxSteps(10)
    eft = PETSc.TS.ExactFinalTime.STEPOVER
    ts.setExactFinalTime(eft)

    x = PETSc.Vec().createSeq(nx*ny*nz)
    ts.setSolution(x)
    app = AppCtx(nx, ny, nz)
    f = PETSc.Vec().createSeq(nx*ny*nz)
    ts.setIFunction(app.formFunction, f)
    ts.snes.setUseMF(1)
    ts.snes.ksp.setType('cg')
    ts.setFromOptions()
    ts.setUp()

    wt = 1e300
    for i in range(samples):
        app.formInitial(0, x)
        t1 = PETSc.Log.getTime()
        ts.solve(x)
        t2 = PETSc.Log.getTime()
        wt = min(wt,t2-t1)
        
    if plot and pylab: 
        app.plot(ts.time, x)

    return wt

OptDB = PETSc.Options()


start = OptDB.getInt('start', 12)
step = OptDB.getInt('step', 4)
stop = OptDB.getInt('stop', start)
samples = OptDB.getInt('samples', 1)

plot = OptDB.getBool('plot', False)
if plot and not pylab:
    PETSc.Sys.Print("matplotlib not available")

for n in range(start, stop+step, step):
    nx = ny = nz = n+1
    wt = run_test(nx,ny,nz,samples,plot)
    PETSc.Sys.Print("Grid %3d x %3d x %3d -> %f  seconds (%2d samples)" 
                    % (nx,ny,nz,wt,samples))

if plot and pylab:
    pylab.show()
