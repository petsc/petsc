# --------------------------------------------------------------------

from petsc4py import PETSc
import unittest
from sys import getrefcount

# --------------------------------------------------------------------

class TestKSPBase(object):

    KSP_TYPE = None
    PC_TYPE  = None

    def setUp(self):
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_SELF)
        if self.KSP_TYPE:
            ksp.setType(self.KSP_TYPE)
        if self.PC_TYPE:
            pc = ksp.getPC()
            pc.setType(self.PC_TYPE)
        self.ksp = ksp

    def tearDown(self):
        self.ksp = None

    def testGetSetType(self):
        self.assertEqual(self.ksp.getType(), self.KSP_TYPE)
        self.ksp.setType(self.KSP_TYPE)
        self.assertEqual(self.ksp.getType(), self.KSP_TYPE)

    def testTols(self):
        tols = self.ksp.getTolerances()
        self.ksp.setTolerances(*tols)
        tnames = ('rtol', 'atol', 'divtol', 'max_it')
        tolvals = [getattr(self.ksp, t) for t in  tnames]
        self.assertEqual(tuple(tols), tuple(tolvals))

    def testGetSetPC(self):
        oldpc = self.ksp.getPC()
        self.assertEqual(oldpc.getRefCount(), 2)
        newpc = PETSc.PC()
        newpc.create(self.ksp.getComm())
        self.assertEqual(newpc.getRefCount(), 1)
        self.ksp.setPC(newpc)
        self.assertEqual(newpc.getRefCount(), 2)
        self.assertEqual(oldpc.getRefCount(), 1)
        oldpc.destroy()
        self.assertFalse(bool(oldpc))
        pc = self.ksp.getPC()
        self.assertTrue(bool(pc))
        self.assertEqual(pc, newpc)
        self.assertEqual(pc.getRefCount(), 3)
        newpc.destroy()
        self.assertFalse(bool(newpc))
        self.assertEqual(pc.getRefCount(), 2)

    def testSolve(self):
        A = PETSc.Mat().create(PETSc.COMM_SELF)
        A.setSizes([3,3])
        A.setType(PETSc.Mat.Type.SEQAIJ)
        for i in range(3):
            A.setValue(i, i, 0.9/(i+1))
        A.assemble()
        A.shift(1)
        x, b = A.getVecs()
        b.set(10)
        x.setRandom()
        self.ksp.setOperators(A)
        self.ksp.setConvergenceHistory()
        self.ksp.solve(b, x)
        rh = self.ksp.getConvergenceHistory()
        self.ksp.setConvergenceHistory(0)
        rh = self.ksp.getConvergenceHistory()
        self.assertEqual(len(rh), 0)
        del A, x, b

    def testSetMonitor(self):
        reshist = {}
        def monitor(ksp, its, rnorm):
            reshist[its] = rnorm
        refcnt = getrefcount(monitor)
        self.ksp.setMonitor(monitor)
        self.assertEqual(getrefcount(monitor), refcnt + 1)
        ## self.testSolve()
        reshist = {}
        self.ksp.cancelMonitor()
        self.assertEqual(getrefcount(monitor), refcnt)
        self.testSolve()
        self.assertEqual(len(reshist), 0)
        ## Monitor = PETSc.KSP.Monitor
        ## self.ksp.setMonitor(Monitor())
        ## self.ksp.setMonitor(Monitor.DEFAULT)
        ## self.ksp.setMonitor(Monitor.TRUE_RESIDUAL_NORM)
        ## self.ksp.setMonitor(Monitor.SOLUTION)

    def testSetConvergenceTest(self):
        def converged(ksp, its, rnorm):
            if its > 10: return True
            return False
        refcnt = getrefcount(converged)
        self.ksp.setConvergenceTest(converged)
        self.assertEqual(getrefcount(converged), refcnt + 1)
        self.ksp.setConvergenceTest(None)
        self.assertEqual(getrefcount(converged), refcnt)

# --------------------------------------------------------------------

class TestKSPPREONLY(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.PREONLY
    PC_TYPE = PETSc.PC.Type.LU

class TestKSPRICHARDSON(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.RICHARDSON

class TestKSPCHEBYCHEV(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.CHEBYCHEV

class TestKSPCG(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.CG

class TestKSPCGNE(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.CGNE

class TestKSPSTCG(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.STCG

class TestKSPBCGS(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.BCGS

class TestKSPBCGSL(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.BCGSL

class TestKSPCGS(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.CGS

class TestKSPQCG(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.QCG
    PC_TYPE  = PETSc.PC.Type.JACOBI

class TestKSPBICG(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.BICG

class TestKSPGMRES(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.GMRES

class TestKSPFGMRES(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.FGMRES

# --------------------------------------------------------------------

class MyKSP(object):

    def __init__(self):
        pass

    def create(self, ksp):
        self.work = []

    def destroy(self):
        for v in self.work:
            v.destroy()

    def setUp(self, ksp):
        self.work[:] = ksp.getWorkVecs(right=2, left=None)

    def loop(self, ksp, r):
        its = ksp.getIterationNumber()
        rnorm = r.norm()
        ksp.setResidualNorm(rnorm)
        ksp.logConvergenceHistory(its, rnorm)
        ksp.callMonitor(its, rnorm)
        reason =  ksp.callConvergenceTest(its, rnorm)
        if reason:
            ksp.setConvergedReason(reason)
        else:
            ksp.setIterationNumber(its+1)
        return reason

class MyRichardson(MyKSP):

    def solve(self, ksp):
        A, B, flag = ksp.getOperators()
        P = ksp.getPC()
        x = ksp.getSolution()
        b = ksp.getRhs()
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

    def solve(self, ksp):
        A, B, flag = ksp.getOperators()
        P = ksp.getPC()
        x = ksp.getSolution()
        b = ksp.getRhs()
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

class TestKSPPYTHONBase(TestKSPBase):

    KSP_TYPE = PETSc.KSP.Type.PYTHON
    ContextClass = None

    def setUp(self):
        super(TestKSPPYTHONBase, self).setUp()
        ctx = self.ContextClass()
        self.ksp.setPythonContext(ctx)

class TestKSPPYTHON_RICH(TestKSPPYTHONBase, unittest.TestCase):
    PC_TYPE  = PETSc.PC.Type.JACOBI
    ContextClass = MyRichardson

class TestKSPPYTHON_CG(TestKSPPYTHONBase, unittest.TestCase):
    PC_TYPE  = PETSc.PC.Type.NONE
    ContextClass = MyCG

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
