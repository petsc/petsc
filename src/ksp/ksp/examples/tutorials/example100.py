from __future__ import print_function

# --------------------------------------------------------------------

from petsc4py import PETSc

# --------------------------------------------------------------------

OptDB = PETSc.Options()

INFO = OptDB.hasName('info')

def LOG(arg):
    if INFO:
        print(arg)

# --------------------------------------------------------------------

class Laplace1D(object):

    def create(self, A):
        LOG('Laplace1D.create()')
        M, N = A.getSize()
        assert M == N

    def destroy(self, A):
        LOG('Laplace1D.destroy()')

    def view(self, A, vw):
        LOG('Laplace1D.view()')

    def setFromOptions(self, A):
        LOG('Laplace1D.setFromOptions()')

    def setUp(self, A):
        LOG('Laplace1D.setUp()')

    def assemblyBegin(self, A, flag):
        LOG('Laplace1D.assemblyBegin()')

    def assemblyEnd(self, A, flag):
        LOG('Laplace1D.assemblyEnd()')

    def getDiagonal(self, A, d):
        LOG('Laplace1D.getDiagonal()')
        M, N = A.getSize()
        h = 1.0/(M-1)
        d.set(2.0/h**2)

    def mult(self, A, x, y):
        LOG('Laplace1D.mult()')
        M, N = A.getSize()
        xx = x.getArray(readonly=1) # to numpy array
        yy = y.getArray(readonly=0) # to numpy array
        yy[0]    =  2.0*xx[0] - xx[1]
        yy[1:-1] = - xx[:-2] + 2.0*xx[1:-1] - xx[2:]
        yy[-1]   = - xx[-2] + 2.0*xx[-1]
        h = 1.0/(M-1)
        yy *= 1.0/h**2

    def multTranspose(self, A, x, y):
        LOG('Laplace1D.multTranspose()')
        self.mult(A, x, y)


# --------------------------------------------------------------------

class Jacobi(object):

    def create(self, pc):
        LOG('Jacobi.create()')
        self.diag = None

    def destroy(self, pc):
        LOG('Jacobi.destroy()')
        if self.diag:
            self.diag.destroy()

    def view(self, pc, vw):
        LOG('Jacobi.view()')

    def setFromOptions(self, pc):
        LOG('Jacobi.setFromOptions()')

    def setUp(self, pc):
        LOG('Jacobi.setUp()')
        A, B = pc.getOperators()
        self.diag = B.getDiagonal(self.diag)

    def apply(self, pc, x, y):
        LOG('Jacobi.apply()')
        y.pointwiseDivide(x, self.diag)

    def applyTranspose(self, pc, x, y):
        LOG('Jacobi.applyTranspose()')
        self.apply(pc, x, y)

# --------------------------------------------------------------------

class ConjGrad(object):

    def create(self, ksp):
        LOG('ConjGrad.create()')
        self.work = []

    def destroy(self, ksp):
        LOG('ConjGrad.destroy()')
        for vec in self.work:
            if vec:
                vec.destroy()
        self.work = []

    def view(self, ksp, viewer):
        LOG('ConjGrad.view()')

    def setUp(self, ksp):
        LOG('ConjGrad.setUp()')
        self.work[:] = ksp.getWorkVecs(right=3, left=None)

    def solve(self, ksp, b, x):
        LOG('ConjGrad.solve()')
        A, P = get_op_pc(ksp, transpose=False)
        pcg(ksp, A, P, b, x, *self.work)

    def solveTranspose(self, ksp, b, x):
        LOG('ConjGrad.solveTranspose()')
        A, P = get_op_pc(ksp, transpose=True)
        pcg(ksp, A, P, b, x, *self.work)

def get_op_pc(ksp, transpose=False):
    op, _ = ksp.getOperators()
    pc = ksp.getPC()
    if not transpose:
        A = op.mult
        P = pc.apply
    else:
        A = op.multTranspose
        P = pc.applyTranspose
    return A, P

def do_loop(ksp, r):
    its = ksp.getIterationNumber()
    rnorm = r.norm()
    ksp.setResidualNorm(rnorm)
    ksp.logConvergenceHistory(rnorm)
    ksp.monitor(its, rnorm)
    reason = ksp.callConvergenceTest(its, rnorm)
    if not reason:
        ksp.setIterationNumber(its+1)
    else:
        ksp.setConvergedReason(reason)
    return reason

def pcg(ksp, A, P, b, x, r, z, p):
    A(x, r)
    r.aypx(-1, b)
    P(r, z)
    delta = r.dot(z)
    z.copy(p)
    while not do_loop(ksp, z):
        A(p, z)
        alpha = delta / z.dot(p)
        x.axpy(+alpha, p)
        r.axpy(-alpha, z)
        P(r, z)
        delta_old = delta
        delta = r.dot(z)
        beta = delta / delta_old
        p.aypx(beta, z)

def richardson(ksp, A, P, b, x, r, z):
    A(x, r)
    r.aypx(-1, b)
    P(r, z)
    x.axpy(1, z)
    while not do_loop(ksp, z):
        A(x, r)
        r.aypx(-1, b)
        P(r, z)
        x.axpy(1, z)

# --------------------------------------------------------------------
