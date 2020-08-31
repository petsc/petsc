# --------------------------------------------------------------------

from petsc4py import PETSc
import unittest
from sys import getrefcount

# --------------------------------------------------------------------

class MyKSP(object):

    def __init__(self):
        pass

    def create(self, ksp):
        self.work = []

    def destroy(self, ksp):
        for v in self.work:
            v.destroy()

    def setUp(self, ksp):
        self.work[:] = ksp.getWorkVecs(right=2, left=None)

    def reset(self, ksp):
        for v in self.work:
            v.destroy()
        del self.work[:]

    def loop(self, ksp, r):
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

class MyRichardson(MyKSP):

    def solve(self, ksp, b, x):
        A, B = ksp.getOperators()
        P = ksp.getPC()
        r, z = self.work
        #
        A.mult(x, r)
        r.aypx(-1, b)
        P.apply(r, z)
        x.axpy(1, z)
        while not self.loop(ksp, z):
            A.mult(x, r)
            r.aypx(-1, b)
            P.apply(r, z)
            x.axpy(1, z)

class MyCG(MyKSP):

    def setUp(self, ksp):
        super(MyCG, self).setUp(ksp)
        d = self.work[0].duplicate()
        q = d.duplicate()
        self.work += [d, q]

    def solve(self, ksp, b, x):
        A, B = ksp.getOperators()
        P = ksp.getPC()
        r, z, d, q = self.work
        #
        A.mult(x, r)
        r.aypx(-1, b)
        r.copy(d)
        delta_0 = r.dot(r)
        delta = delta_0
        while not self.loop(ksp, r):
            A.mult(d, q)
            alpha = delta / d.dot(q)
            x.axpy(+alpha, d)
            r.axpy(-alpha, q)
            delta_old = delta
            delta = r.dot(r)
            beta = delta / delta_old
            d.aypx(beta, r)

# --------------------------------------------------------------------

from test_ksp import BaseTestKSP

class BaseTestKSPPYTHON(BaseTestKSP):

    KSP_TYPE = PETSc.KSP.Type.PYTHON
    ContextClass = None

    def setUp(self):
        super(BaseTestKSPPYTHON, self).setUp()
        ctx = self.ContextClass()
        self.ksp.setPythonContext(ctx)

class TestKSPPYTHON_RICH(BaseTestKSPPYTHON, unittest.TestCase):
    PC_TYPE  = PETSc.PC.Type.JACOBI
    ContextClass = MyRichardson

class TestKSPPYTHON_CG(BaseTestKSPPYTHON, unittest.TestCase):
    PC_TYPE  = PETSc.PC.Type.NONE
    ContextClass = MyCG

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
