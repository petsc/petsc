import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import math

class MyODE:
  def __init__(self,da):
    self.da = da
  def function(self, ts,t,x,f):
    mx = da.getSizes(); mx = mx[0]; hx = 1.0/mx
    (xs,xm) = da.getCorners(); xs = xs[0]; xm = xm[0]
    xx = da.createLocalVector()
    da.globalToLocal(x,xx)
    dt = ts.getTimeStep()
    x0 = ts.getSolution()
    if xs == 0: f[0] = xx[0]/hx; xs = 1;
    if xs+xm >= mx: f[mx-1] = xx[xm-(xs==1)]/hx; xm = xm-(xs==1);
    for i in range(xs,xs+xm-1):
      f[i] = (xx[i-xs+1] - x0[i])/dt + (2.0*xx[i-xs+1] - xx[i-xs] - xx[i-xs+2])/hx - hx*math.exp(xx[i-xs+1])
    f.assemble()
  def jacobian(self,ts,t,x,J,P):
    mx = da.getSizes(); mx = mx[0]; hx = 1.0/mx
    (xs,xm) = da.getCorners(); xs = xs[0]; xm = xm[0]
    xx = da.createLocalVector()
    da.globalToLocal(x,xx)
    x0 = ts.getSolution()
    dt = ts.getTimeStep()
    P.zeroEntries()
    if xs == 0: P.setValues([0],[0],1.0/hx); xs = 1;
    if xs+xm >= mx: P.setValues([mx-1],[mx-1],1.0/hx); xm = xm-(xs==1);
    for i in range(xs,xs+xm-1):
      P.setValues([i],[i-1,i,i+1],[-1.0/hx,1.0/dt+2.0/hx-hx*math.exp(xx[i-xs+1]),-1.0/hx])
    P.assemble()
    return True # same_nz

da = PETSc.DA().create([9],comm=PETSc.COMM_WORLD)
f = da.createGlobalVector()
x = f.duplicate()
J = da.getMatrix(PETSc.Mat.Type.AIJ);
   
ts = PETSc.TS().create(PETSc.COMM_WORLD)
ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)
ts.setType('python')

ode = MyODE(da)
ts.setFunction(ode.function, f)
ts.setJacobian(ode.jacobian, J, J)

ts.setTimeStep(0.1)
ts.setDuration(10, 1.0)
ts.setFromOptions()
x.set(1.0)
ts.solve(x)

