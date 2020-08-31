import unittest
from petsc4py import PETSc

# --------------------------------------------------------------------

class TestVersion(unittest.TestCase):

    def testGetVersion(self):
        version = PETSc.Sys.getVersion()
        self.assertTrue(version > (0, 0, 0))
        v, date = PETSc.Sys.getVersion(date=True)
        self.assertTrue(version == v)
        self.assertTrue(isinstance(date, str))
        v, author = PETSc.Sys.getVersion(author=True)
        self.assertTrue(version == v)
        self.assertTrue(isinstance(author, (list,tuple)))

    def testGetVersionInfo(self):
        version = PETSc.Sys.getVersion()
        info = PETSc.Sys.getVersionInfo()
        self.assertEqual(version,
                         (info['major'],
                          info['minor'],
                          info['subminor'],))
        self.assertTrue(isinstance(info['release'], bool))
        v, date = PETSc.Sys.getVersion(date=True)
        self.assertEqual(date, info['date'])

    def testGetSetDefaultComm(self):
        c = PETSc.Sys.getDefaultComm()
        self.assertEqual(c, PETSc.COMM_WORLD)
        PETSc.Sys.setDefaultComm(PETSc.COMM_SELF)
        c = PETSc.Sys.getDefaultComm()
        self.assertEqual(c, PETSc.COMM_SELF)
        PETSc.Sys.setDefaultComm(PETSc.COMM_WORLD)
        c = PETSc.Sys.getDefaultComm()
        self.assertEqual(c, PETSc.COMM_WORLD)
        f = lambda : PETSc.Sys.setDefaultComm(PETSc.COMM_NULL)
        self.assertRaises(ValueError, f)

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

# --------------------------------------------------------------------
