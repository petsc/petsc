# Stiff 3-variable ODE system from chemical reactions,
# due to Robertson (1966),
# problem ROBER in Hairer&Wanner, ODE 2, 1996

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

class Rober(object):
    n = 3
    comm = PETSc.COMM_SELF
    def evalSolution(self, t, x):
        assert t == 0.0, "only for t=0.0"
        x[:] = [1, 0, 0]
        x.assemble()
    def evalFunction(self, ts, t, x, xdot, a, f):
        f[:] = [xdot[0] + 0.04*x[0] - 1e4*x[1]*x[2],
                xdot[1] - 0.04*x[0] + 1e4*x[1]*x[2] + 3e7*x[1]**2,
                xdot[2] - 3e7*x[1]**2]
        f.assemble()
    def evalJacobian(self, ts, t, x, xdot, a, A, B):
        J = B
        J[:,:] = [[a + 0.04, -1e4*x[2],                -1e4*x[1]],
                  [-0.04,    a + 1e4*x[2] + 3e7*2*x[1], 1e4*x[1]],
                  [0,        -3e7*2*x[1],               a]]
        J.assemble()
        if A != B: A.assemble()
        return True # same nonzero pattern

OptDB = PETSc.Options()
ode = Rober()

J = PETSc.Mat().createDense([ode.n, ode.n], comm=ode.comm)
x = PETSc.Vec().createSeq(ode.n, comm=ode.comm)

ts = PETSc.TS().create(comm=ode.comm)
ts.setProblemType(ts.ProblemType.NONLINEAR)
ts.setType(ts.Type.THETA)

ts.setIFunction(ode.evalFunction)
ts.setIJacobian(ode.evalJacobian, J)

ts.setTime(0.0)
ts.setTimeStep(.001)
ts.setMaxTime(1e30)
ts.setMaxSteps(100)

ts.setFromOptions()
ode.evalSolution(0.0, x)
ts.solve(x)

del ode, J, x, ts
