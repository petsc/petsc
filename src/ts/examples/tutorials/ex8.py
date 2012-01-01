import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import math

class MyODE:
  def __init__(self,da):
    self.da = da
  def getCorners(self):
    'get corners and ghost corners, take first element of array because this is a 1D problem'
    xs, xm = self.da.getCorners()
    gxs, gxm = self.da.getGhostCorners()
    return xs[0], xm[0], gxs[0], gxm[0]
  def function(self, ts,t,x,xdot,f):
    mx = da.getSizes(); mx = mx[0]; hx = 1.0/mx
    (xs,xm,gxs,gxm) = self.getCorners()
    xx = da.createLocalVector()
    xxdot = da.createLocalVector()
    da.globalToLocal(x,xx)
    da.globalToLocal(xdot,xxdot)
    dt = ts.getTimeStep()
    x0 = ts.getSolution()
    lxs, lxe = xs, xs+xm
    if xs == 0: f[0] = xx[0]/hx; lxs+=1;
    if lxe == mx: f[mx-1] = xx[mx-1-gxs]/hx; lxe-=1
    for i in range(lxs,lxe):
      f[i] = xxdot[i-gxs] + (2.0*xx[i-gxs] - xx[i-1-gxs] - xx[i+1-gxs])/hx - hx*math.exp(xx[i-gxs])
    f.assemble()
  def jacobian(self,ts,t,x,xdot,shift,J,P):
    mx = da.getSizes(); mx = mx[0]; hx = 1.0/mx
    (xs,xm,gxs,gxm) = self.getCorners()
    xx = da.createLocalVector()
    da.globalToLocal(x,xx)
    x0 = ts.getSolution()
    dt = ts.getTimeStep()
    P.zeroEntries()
    lxs, lxe = xs, xs+xm
    if xs == 0: P.setValues([0],[0],1.0/hx); lxs+=1
    if lxe == mx: P.setValues([mx-1],[mx-1],1.0/hx); lxe-=1
    for i in range(lxs,lxe):
      P.setValues([i],[i-1,i,i+1],[-1.0/hx,2.0/hx-hx*math.exp(xx[i-gxs])+shift,-1.0/hx])
    P.assemble()
    return True # same_nz

M = PETSc.Options().getInt('M', 9)
da = PETSc.DA().create([M],comm=PETSc.COMM_WORLD)
f = da.createGlobalVector()
x = f.duplicate()
J = da.getMatrix(PETSc.Mat.Type.AIJ);
   
ts = PETSc.TS().create(PETSc.COMM_WORLD)
ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)
ts.setType(ts.Type.GL)

ode = MyODE(da)
ts.setIFunction(ode.function, f)
ts.setIJacobian(ode.jacobian, J)

ts.setTimeStep(0.1)
ts.setDuration(10, 1.0)
ts.setFromOptions()
x.set(1.0)
ts.solve(x)
