import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

class BouncingBall(object):
    n = 2
    comm = PETSc.COMM_SELF

    def __init__(self):
        pass

    def initialCondition(self, u):
        u[0] = 0.0
        u[1] = 20.0
        u.assemble()

    def evalRHSFunction(self, ts, t, u, f):
        f[0] = u[1]
        f[1] = -9.8
        f.assemble()

    def evalRHSJacobian(self, ts, t, u, A, B):
        J = A
        J[0,0] = 0.0
        J[1,0] = 0.0
        J[0,1] = 1.0
        J[1,1] = 0.0
        J.assemble()
        if A != B: B.assemble()
        return True # same nonzero pattern

class Monitor(object):

    def __init__(self):
        pass

    def __call__(self, ts, k, t, x):
        PETSc.Sys.Print(f"Position at time {t}: {x[0]}")

ode = BouncingBall()

J = PETSc.Mat().create()
J.setSizes([ode.n,ode.n])
J.setType('aij')
J.setUp()
J.assemble()

u = PETSc.Vec().createSeq(ode.n, comm=ode.comm)
f = u.duplicate()

ts = PETSc.TS().create(comm=ode.comm)
ts.setProblemType(ts.ProblemType.NONLINEAR)

ts.setType(ts.Type.BEULER)
ts.setRHSFunction(ode.evalRHSFunction, f)
ts.setRHSJacobian(ode.evalRHSJacobian, J)

#ts.setSaveTrajectory()
ts.setTime(0.0)
ts.setTimeStep(0.01)
ts.setMaxTime(15.0)
#ts.setMaxSteps(1000)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
#ts.setMonitor(Monitor())

direction = [-1]
terminate = [False]

def event(ts, t, X, fvalue):
    fvalue[0] = X[0]

def postevent(ts, events, t, X, forward):
    X[0] = 0.0
    X[1] = -0.9*X[1]
    X.assemble()

ts.setEventHandler(direction, terminate, event, postevent)
ts.setEventTolerances(1e-6, vtol=[1e-9])

ts.setFromOptions()

ode.initialCondition(u)
ts.solve(u)

