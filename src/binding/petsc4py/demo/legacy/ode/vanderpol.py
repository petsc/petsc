# Testing TSAdjoint and matrix-free Jacobian
# Basic usage:
#     python vanderpol.py
# Test implicit methods using implicit form:
#     python -implicitform
# Test explicit methods:
#     python -implicitform 0
# Test IMEX methods:
#     python -imexform
# Matrix-free implementations can be enabled with an additional option -mf

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

class VDP(object):
    n = 2
    comm = PETSc.COMM_SELF
    def __init__(self, mu_=1.0e3, mf_=False, imex_=False):
        self.mu_ = mu_
        self.mf_ = mf_
        self.imex_ = imex_
        if self.mf_:
            self.Jim_ = PETSc.Mat().createDense([self.n,self.n], comm=self.comm)
            self.Jim_.setUp()
            self.JimP_ = PETSc.Mat().createDense([self.n,1], comm=self.comm)
            self.JimP_.setUp()
            self.Jex_ = PETSc.Mat().createDense([self.n,self.n], comm=self.comm)
            self.Jex_.setUp()
            self.JexP_ = PETSc.Mat().createDense([self.n,1], comm=self.comm)
            self.JexP_.setUp()
    def initialCondition(self, u):
        mu = self.mu_
        u[0] = 2.0
        u[1] = -2.0/3.0 + 10.0/(81.0*mu) - 292.0/(2187.0*mu*mu)
        u.assemble()
    def evalFunction(self, ts, t, u, f):
        mu = self.mu_
        f[0] = u[1]
        if self.imex_:
            f[1] = 0.0
        else:
            f[1] = mu*((1.-u[0]*u[0])*u[1]-u[0])
        f.assemble()
    def evalJacobian(self, ts, t, u, A, B):
        if not self.mf_:
            J = A
        else :
            J = self.Jex_
        mu = self.mu_
        J[0,0] = 0
        J[0,1] = 1.0
        if self.imex_:
            J[1,0] = 0
            J[1,1] = 0
        else:
            J[1,0] = -mu*(2.0*u[1]*u[0]+1.)
            J[1,1] = mu*(1.0-u[0]*u[0])
        J.assemble()
        if A != B: B.assemble()
        return True # same nonzero pattern
    def evalJacobianP(self, ts, t, u, C):
        if not self.mf_:
            Jp = C
        else:
            Jp = self.JexP_
        if not self.imex_:
            Jp[0,0] = 0
            Jp[1,0] = (1.-u[0]*u[0])*u[1]-u[0]
            Jp.assemble()
        return True
    def evalIFunction(self, ts, t, u, udot, f):
        mu = self.mu_
        if self.imex_:
            f[0] = udot[0]
        else:
            f[0] = udot[0]-u[1]
        f[1] = udot[1]-mu*((1.-u[0]*u[0])*u[1]-u[0])
        f.assemble()
    def evalIJacobian(self, ts, t, u, udot, shift, A, B):
        if not self.mf_:
            J = A
        else :
            J = self.Jim_
        mu = self.mu_
        if self.imex_:
            J[0,0] = shift
            J[0,1] = 0.0
        else:
            J[0,0] = shift
            J[0,1] = -1.0
        J[1,0] = mu*(2.0*u[1]*u[0]+1.)
        J[1,1] = shift-mu*(1.0-u[0]*u[0])
        J.assemble()
        if A != B: B.assemble()
        return True # same nonzero pattern
    def evalIJacobianP(self, ts, t, u, udot, shift, C):
        if not self.mf_:
            Jp = C
        else:
            Jp = self.JimP_
        Jp[0,0] = 0
        Jp[1,0] = u[0]-(1.-u[0]*u[0])*u[1]
        Jp.assemble()
        return True

class JacShell:
    def __init__(self, ode):
        self.ode_ = ode
    def mult(self, A, x, y):
        "y <- A * x"
        self.ode_.Jex_.mult(x,y)
    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.ode_.Jex_.multTranspose(x, y)

class JacPShell:
    def __init__(self, ode):
        self.ode_ = ode
    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.ode_.JexP_.multTranspose(x, y)

class IJacShell:
    def __init__(self, ode):
        self.ode_ = ode
    def mult(self, A, x, y):
        "y <- A * x"
        self.ode_.Jim_.mult(x,y)
    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.ode_.Jim_.multTranspose(x, y)

class IJacPShell:
    def __init__(self, ode):
        self.ode_ = ode
    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.ode_.JimP_.multTranspose(x, y)

OptDB = PETSc.Options()

mu_ = OptDB.getScalar('mu', 1.0e3)
mf_ = OptDB.getBool('mf', False)

implicitform_ = OptDB.getBool('implicitform', False)
imexform_ = OptDB.getBool('imexform', False)

ode = VDP(mu_,mf_,imexform_)

if not mf_:
    Jim = PETSc.Mat().createDense([ode.n,ode.n], comm=ode.comm)
    Jim.setUp()
    JimP = PETSc.Mat().createDense([ode.n,1], comm=ode.comm)
    JimP.setUp()
    Jex = PETSc.Mat().createDense([ode.n,ode.n], comm=ode.comm)
    Jex.setUp()
    JexP = PETSc.Mat().createDense([ode.n,1], comm=ode.comm)
    JexP.setUp()
else:
    Jim = PETSc.Mat().create()
    Jim.setSizes([ode.n,ode.n])
    Jim.setType('python')
    shell = IJacShell(ode)
    Jim.setPythonContext(shell)
    Jim.setUp()
    Jim.assemble()
    JimP = PETSc.Mat().create()
    JimP.setSizes([ode.n,1])
    JimP.setType('python')
    shell = IJacPShell(ode)
    JimP.setPythonContext(shell)
    JimP.setUp()
    JimP.assemble()
    Jex = PETSc.Mat().create()
    Jex.setSizes([ode.n,ode.n])
    Jex.setType('python')
    shell = JacShell(ode)
    Jex.setPythonContext(shell)
    Jex.setUp()
    Jex.assemble()
    JexP = PETSc.Mat().create()
    JexP.setSizes([ode.n,1])
    JexP.setType('python')
    shell = JacPShell(ode)
    JexP.setPythonContext(shell)
    JexP.setUp()
    JexP.zeroEntries()
    JexP.assemble()

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

if imexform_:
    ts.setType(ts.Type.ARKIMEX)
    ts.setIFunction(ode.evalIFunction, f)
    ts.setIJacobian(ode.evalIJacobian, Jim)
    ts.setIJacobianP(ode.evalIJacobianP, JimP)
    ts.setRHSFunction(ode.evalFunction, f)
    ts.setRHSJacobian(ode.evalJacobian, Jex)
    ts.setRHSJacobianP(ode.evalJacobianP, JexP)
else:
    if implicitform_:
        ts.setType(ts.Type.CN)
        ts.setIFunction(ode.evalIFunction, f)
        ts.setIJacobian(ode.evalIJacobian, Jim)
        ts.setIJacobianP(ode.evalIJacobianP, JimP)
    else:
        ts.setType(ts.Type.RK)
        ts.setRHSFunction(ode.evalFunction, f)
        ts.setRHSJacobian(ode.evalJacobian, Jex)
        ts.setRHSJacobianP(ode.evalJacobianP, JexP)

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

del ode, Jim, JimP, Jex, JexP, u, f, ts, adj_u, adj_p
