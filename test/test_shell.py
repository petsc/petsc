from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------

class BaseTestShell(object):
    COMM = None
    def setUp(self):
        self.shell = PETSc.Shell().create(self.COMM)
    def tearDown(self):
        self.shell = None
    def testDEFAULT(self):
        shell = PETSc.Shell.DEFAULT(self.COMM)
        self.assertNotEqual(self.shell, shell)
        shell = None

class TestShell(BaseTestShell, unittest.TestCase):
    pass

class TestShellSELF(BaseTestShell, unittest.TestCase):
    COMM = PETSc.COMM_SELF

class TestShellWORLD(BaseTestShell, unittest.TestCase):
    COMM = PETSc.COMM_WORLD

if PETSc.Sys.getVersion() < (3,2,0):
    del BaseTestShell
    del TestShell
    del TestShellSELF
    del TestShellWORLD

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
