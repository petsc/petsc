import unittest

from petsc4py import PETSc


class TestDMComposite(unittest.TestCase):
    COMM = PETSc.COMM_WORLD

    def setUp(self):
        self.dms = [PETSc.DMDA().create([n], comm=self.COMM) for n in (3, 4)]
        self.dm = PETSc.DMComposite().create(self.COMM)
        self.dm.addDM(*self.dms)
        self.vec = self.dm.createGlobalVec()

    def tearDown(self):
        self.vec.destroy()
        self.dm.destroy()
        for dm in self.dms:
            dm.destroy()
        PETSc.garbage_cleanup()

    def testAccessReferenceCounts(self):
        for locs in (None, [1]):
            with self.subTest(locs=locs):
                for _ in range(3):
                    with self.dm.getAccess(self.vec, locs) as vecs:
                        for vec in vecs:
                            self.assertEqual(vec.getRefCount(), 2)
                            vec.set(2)
                    for vec in vecs:
                        self.assertFalse(vec)
        self.assertEqual(self.vec.sum(), 14)

    def testAccessException(self):
        with self.assertRaisesRegex(RuntimeError, 'access failed'):
            with self.dm.getAccess(self.vec) as vecs:
                raise RuntimeError('access failed')
        for vec in vecs:
            self.assertFalse(vec)
        with self.dm.getAccess(self.vec) as vecs:
            for vec in vecs:
                self.assertEqual(vec.getRefCount(), 2)

    def testAccessDestroyedWrapper(self):
        with self.dm.getAccess(self.vec) as vecs:
            for vec in vecs:
                vec.destroy()
        with self.dm.getAccess(self.vec) as vecs:
            for vec in vecs:
                self.assertEqual(vec.getRefCount(), 2)


if __name__ == '__main__':
    unittest.main()
