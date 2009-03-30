from petsc4py import PETSc
import unittest
import gc, weakref

# --------------------------------------------------------------------

## gc.set_debug((gc.DEBUG_STATS |
##               gc.DEBUG_LEAK) &
##              ~gc.DEBUG_SAVEALL)

# --------------------------------------------------------------------

class BaseTestGC(object):

    def setUp(self):
        self.obj = self.CLASS().create(comm=PETSc.COMM_SELF)

    def tearDown(self):
        wref = weakref.ref(self.obj)
        self.assertTrue(wref() is self.obj)
        self.obj = None
        gc.collect()
        self.assertTrue(wref() is None)

    def testCycleInSelf(self):
        self.obj.getDict()['self'] = self.obj

    def testCycleInMeth(self):
        self.obj.getDict()['meth'] = self.obj.view

    def testCycleInInst(self):
        class A: pass
        a = A()
        a.obj = self.obj
        self.obj.getDict()['inst'] = a

    def testCycleInManyWays(self):
        self.testCycleInSelf()
        self.testCycleInMeth()
        self.testCycleInInst()

# --------------------------------------------------------------------


class TestGCVec(BaseTestGC, unittest.TestCase):
    CLASS = type('_Vec', (PETSc.Vec,), {})

class TestGCMat(BaseTestGC, unittest.TestCase):
    CLASS = type('_Mat', (PETSc.Mat,), {})

class TestGCKSP(BaseTestGC, unittest.TestCase):
    CLASS = type('_KSP', (PETSc.KSP,), {})

class TestGCSNES(BaseTestGC, unittest.TestCase):
    CLASS = type('_SNES', (PETSc.SNES,), {})
    #def testCycleInAppCtx(self):
    #    self.obj.setAppCtx(self.obj)

class TestGCTS(BaseTestGC, unittest.TestCase):
    CLASS = type('_TS', (PETSc.TS,), {})
    #def testCycleInAppCtx(self):
    #    self.obj.setAppCtx(self.obj)

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
