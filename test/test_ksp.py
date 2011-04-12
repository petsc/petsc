# --------------------------------------------------------------------

from petsc4py import PETSc
import unittest
from sys import getrefcount

# --------------------------------------------------------------------

class BaseTestKSP(object):

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

    def testProperties(self):
        ksp = self.ksp
        #
        ksp.appctx = (1,2,3)
        self.assertEqual(ksp.appctx, (1,2,3))
        ksp.appctx = None
        self.assertEqual(ksp.appctx, None)
        #
        side_orig = ksp.pc_side
        for ps_name in ('LEFT',
                        'RIGHT',
                        'SYMMETRIC'):
            ps_value = getattr(PETSc.PC.Side, ps_name)
            ksp.pc_side = ps_value
            self.assertEqual(ksp.pc_side, ps_value)
        ksp.pc_side = side_orig
        self.assertEqual(ksp.pc_side, side_orig)
        #
        nt_orig = ksp.norm_type
        for nt_name in ('NONE',
                        'PRECONDITIONED',
                        'UNPRECONDITIONED',
                        'NATURAL'):
            nt_value = getattr(PETSc.KSP.NormType, nt_name)
            ksp.norm_type = nt_value
            self.assertEqual(ksp.norm_type, nt_value)
        ksp.norm_type = nt_orig
        self.assertEqual(ksp.norm_type, nt_orig)
        #
        ksp.its = 1
        self.assertEqual(ksp.its, 1)
        ksp.its = 0
        self.assertEqual(ksp.its, 0)
        #
        ksp.norm = 1
        self.assertEqual(ksp.norm, 1)
        ksp.norm = 0
        self.assertEqual(ksp.norm, 0)
        #
        rh = ksp.history
        self.assertTrue(len(rh)==0)
        #
        reason = PETSc.KSP.ConvergedReason.CONVERGED_ITS
        ksp.reason = reason
        self.assertEqual(ksp.reason, reason)
        self.assertTrue(ksp.converged)
        self.assertFalse(ksp.diverged)
        self.assertFalse(ksp.iterating)
        reason = PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT
        ksp.reason = reason
        self.assertEqual(ksp.reason, reason)
        self.assertFalse(ksp.converged)
        self.assertTrue(ksp.diverged)
        self.assertFalse(ksp.iterating)
        reason = PETSc.KSP.ConvergedReason.CONVERGED_ITERATING
        ksp.reason = reason
        self.assertEqual(ksp.reason, reason)
        self.assertFalse(ksp.converged)
        self.assertFalse(ksp.diverged)
        self.assertTrue(ksp.iterating)

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
        x, b = A.createVecs()
        b.set(10)
        x.setRandom()
        self.ksp.setOperators(A)
        self.ksp.setConvergenceHistory()
        self.ksp.solve(b, x)
        r = b.duplicate()
        u = x.duplicate()
        self.ksp.buildSolution(u)
        self.ksp.buildResidual(u)
        rh = self.ksp.getConvergenceHistory()
        self.ksp.setConvergenceHistory(0)
        rh = self.ksp.getConvergenceHistory()
        self.assertEqual(len(rh), 0)
        del A, x, b

    def testResetAndSolve(self):
        self.ksp.reset()
        self.testSolve()
        self.ksp.reset()
        self.testSolve()
        self.ksp.reset()

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

class TestKSPPREONLY(BaseTestKSP, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.PREONLY
    PC_TYPE = PETSc.PC.Type.LU

class TestKSPRICHARDSON(BaseTestKSP, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.RICHARDSON

class TestKSPCHEBYCHEV(BaseTestKSP, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.CHEBYCHEV

class TestKSPCG(BaseTestKSP, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.CG

class TestKSPCGNE(BaseTestKSP, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.CGNE

class TestKSPSTCG(BaseTestKSP, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.STCG

class TestKSPBCGS(BaseTestKSP, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.BCGS

class TestKSPBCGSL(BaseTestKSP, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.BCGSL

class TestKSPCGS(BaseTestKSP, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.CGS

class TestKSPQCG(BaseTestKSP, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.QCG
    PC_TYPE  = PETSc.PC.Type.JACOBI

class TestKSPBICG(BaseTestKSP, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.BICG

class TestKSPGMRES(BaseTestKSP, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.GMRES

class TestKSPFGMRES(BaseTestKSP, unittest.TestCase):
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

    def reset(self, ksp):
        for v in self.work:
            v.destroy()
        del self.work[:]

    def loop(self, ksp, r):
        its = ksp.getIterationNumber()
        rnorm = r.norm()
        ksp.setResidualNorm(rnorm)
        ksp.logConvergenceHistory(its, rnorm)
        ksp.callMonitor(its, rnorm)
        reason = ksp.callConvergenceTest(its, rnorm)
        if not reason:
            ksp.setIterationNumber(its+1)
        else:
            ksp.setConvergedReason(reason)
        return reason

class MyRichardson(MyKSP):

    def solve(self, ksp, b, x):
        A, B, flag = ksp.getOperators()
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
        A, B, flag = ksp.getOperators()
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

try:
    import numpy
    if issubclass(PETSc.ScalarType, numpy.complexfloating):
        del TestKSPSTCG
except:
    pass

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
