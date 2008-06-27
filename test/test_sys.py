import unittest
from petsc4py import PETSc

# --------------------------------------------------------------------

class TestVersion(unittest.TestCase):

    def testGetVersion(self):
        version = PETSc.Sys.getVersion()
        self.assertTrue(version > (0, 0, 0))
        v, patch = PETSc.Sys.getVersion(patch=True)
        self.assertTrue(version == v)
        self.assertTrue(patch >= 0)
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
        _, patch = PETSc.Sys.getVersion(patch=True)
        self.assertEqual(patch, info['patch'])
        v, date = PETSc.Sys.getVersion(date=True)
        self.assertEqual(date, info['date'])

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
