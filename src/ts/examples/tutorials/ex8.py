import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

class MyODE:
    def function(self, ts,t,u,F):
        # print 'MyODE.function()'
        dt = ts.getTimeStep()
        u0 = ts.getSolution()
        f = (u - u0)/dt + u * u
        f.copy(F)
    def jacobian(self,ts,t,u,J,P):
        # print 'MyODE.jacobian()'
        u0 = ts.getSolution()
        dt = ts.getTimeStep()
        P.zeroEntries()
        diag = 1/dt + 2 * u
        P.setDiagonal(diag)
        P.assemble()
        if J != P: J.assemble()
        return True # same_nz

da = PETSc.DA().create([9],comm=PETSc.COMM_WORLD)
f = da.createGlobalVector()
u = f.duplicate()
J = da.getMatrix(PETSc.MatType.AIJ);
    
ts = PETSc.TS().create(PETSc.COMM_WORLD)
ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)
ts.setType('python')

ode = MyODE()
ts.setFunction(ode.function, f)
ts.setJacobian(ode.jacobian, J, J)

ts.setTimeStep(0.1)
ts.setDuration(1.0, 10)
ts.setFromOptions()
u.set(1.0)
ts.solve(u)

