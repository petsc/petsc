from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------

class BaseTestFwk(object):
    COMM = None
    def setUp(self):
        self.fwk = PETSc.Fwk().create(self.COMM)
    def tearDown(self):
        self.fwk = None
    def testDEFAULT(self):
        fwk = PETSc.Fwk.DEFAULT(self.COMM)
        self.assertNotEqual(self.fwk, fwk)
        fwk = None

class TestFwk(BaseTestFwk, unittest.TestCase):
    pass

class TestFwkSELF(BaseTestFwk, unittest.TestCase):
    COMM = PETSc.COMM_SELF

class TestFwkWORLD(BaseTestFwk, unittest.TestCase):
    COMM = PETSc.COMM_WORLD

if PETSc.Sys.getVersion() < (3,2,0):
    del BaseTestFwk
    del TestFwk
    del TestFwkSELF
    del TestFwkWORLD

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
