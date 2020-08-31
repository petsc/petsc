# Testing TSAdjoint and matrix-free Jacobian

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

class VDP(object):
    n = 2
    comm = PETSc.COMM_SELF
    def __init__(self, mu_=1.0e3,mf_=False):
        self.mu_ = mu_
        self.mf_ = mf_
        if self.mf_:
            self.J_ = PETSc.Mat().createDense([self.n,self.n], comm=self.comm)
            self.J_.setUp()
            self.Jp_ = PETSc.Mat().createDense([self.n,1], comm=self.comm)
            self.Jp_.setUp()
    def initialCondition(self, u):
        mu = self.mu_
        u[0] = 2.0
        u[1] = -2.0/3.0 + 10.0/(81.0*mu) - 292.0/(2187.0*mu*mu)
        u.assemble()
    def evalFunction(self, ts, t, u, f):
        mu = self.mu_
        f[0] = u[1]
        f[1] = mu*((1.-u[0]*u[0])*u[1]-u[0])
        f.assemble()
    def evalJacobian(self, ts, t, u, A, B):
        if not self.mf_:
            J = A
        else :
            J = self.J_
        mu = self.mu_
        J[0,0] = 0
        J[1,0] = -mu*(2.0*u[1]*u[0]+1.)
        J[0,1] = 1.0
        J[1,1] = mu*(1.0-u[0]*u[0])
        J.assemble()
        if A != B: B.assemble()
        return True # same nonzero pattern
    def evalJacobianP(self, ts, t, u, C):
        if not self.mf_:
            Jp = C
        else:
            Jp = self.Jp_
        Jp[0,0] = 0
        Jp[1,0] = (1.-u[0]*u[0])*u[1]-u[0]
        Jp.assemble()
        return True
    def evalIFunction(self, ts, t, u, udot, f):
        mu = self.mu_
        f[0] = udot[0]-u[1]
        f[1] = udot[1]-mu*((1.-u[0]*u[0])*u[1]-u[0])
        f.assemble()
    def evalIJacobian(self, ts, t, u, udot, shift, A, B):
        if not self.mf_:
            J = A
        else :
            J = self.J_
        mu = self.mu_
        J[0,0] = shift
        J[1,0] = mu*(2.0*u[1]*u[0]+1.)
        J[0,1] = -1.0
        J[1,1] = shift-mu*(1.0-u[0]*u[0])
        J.assemble()
        if A != B: B.assemble()
        return True # same nonzero pattern
    def evalIJacobianP(self, ts, t, u, udot, shift, C):
        if not self.mf_:
            Jp = C
        else:
            Jp = self.Jp_
        Jp[0,0] = 0
        Jp[1,0] = u[0]-(1.-u[0]*u[0])*u[1]
        Jp.assemble()
        return True

class JacShell:
    def __init__(self, ode):
        self.ode_ = ode
    def mult(self, A, x, y):
        "y <- A * x"
        self.ode_.J_.mult(x,y)
    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.ode_.J_.multTranspose(x, y)

class JacPShell:
    def __init__(self, ode):
        self.ode_ = ode
    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.ode_.Jp_.multTranspose(x, y)
OptDB = PETSc.Options()

mu_ = OptDB.getScalar('mu', 1.0e3)
mf_ = OptDB.getBool('mf', False)

implicitform_ = OptDB.getBool('implicitform', False)

ode = VDP(mu_,mf_)

if not mf_:
    J = PETSc.Mat().createDense([ode.n,ode.n], comm=ode.comm)
    J.setUp()
    Jp = PETSc.Mat().createDense([ode.n,1], comm=ode.comm)
    Jp.setUp()
else:
    J = PETSc.Mat().create()
    J.setSizes([ode.n,ode.n])
    J.setType('python')
    shell = JacShell(ode)
    J.setPythonContext(shell)
    J.setUp()
    J.assemble()
    Jp = PETSc.Mat().create()
    Jp.setSizes([ode.n,1])
    Jp.setType('python')
    shell = JacPShell(ode)
    Jp.setPythonContext(shell)
    Jp.setUp()
    Jp.assemble()

u = PETSc.Vec().createSeq(ode.n, comm=ode.comm)
f = u.duplicate()
adj_u = []
adj_u.append(PETSc.Vec().createSeq(ode.n, comm=ode.comm))
adj_u.append(PETSc.Vec().createSeq(ode.n, comm=ode.comm))
adj_p = []
adj_p.append(PETSc.Vec().createSeq(1, comm=ode.comm))
adj_p.append(PETSc.Vec().createSeq(1, comm=ode.comm))

ts = PETSc.TS().create(comm=ode.comm)
ts.setProblemType(ts.ProblemType.NONLINEAR)

if implicitform_:
    ts.setType(ts.Type.CN)
    ts.setIFunction(ode.evalIFunction, f)
    ts.setIJacobian(ode.evalIJacobian, J)
    ts.setIJacobianP(ode.evalIJacobianP, Jp)
else:
    ts.setType(ts.Type.RK)
    ts.setRHSFunction(ode.evalFunction, f)
    ts.setRHSJacobian(ode.evalJacobian, J)
    ts.setRHSJacobianP(ode.evalJacobianP, Jp)

ts.setSaveTrajectory()
ts.setTime(0.0)
ts.setTimeStep(0.001)
ts.setMaxTime(0.5)
ts.setMaxSteps(1000)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

ts.setFromOptions()
ode.initialCondition(u)
ts.solve(u)

adj_u[0][0] = 1
adj_u[0][1] = 0
adj_u[0].assemble()
adj_u[1][0] = 0
adj_u[1][1] = 1
adj_u[1].assemble()
adj_p[0][0] = 0
adj_p[0].assemble()
adj_p[1][0] = 0
adj_p[1].assemble()

ts.setCostGradients(adj_u,adj_p)

ts.adjointSolve()

adj_u[0].view()
adj_u[1].view()
adj_p[0].view()
adj_p[1].view()

def compute_derp(du,dp):
    print(du[1]*(-10.0/(81.0*mu_*mu_)+2.0*292.0/(2187.0*mu_*mu_*mu_))+dp[0])

compute_derp(adj_u[0],adj_p[0])
compute_derp(adj_u[1],adj_p[1])

del ode, J, Jp, u, f, ts, adj_u, adj_p
