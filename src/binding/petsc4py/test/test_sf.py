import unittest

import numpy as np

from petsc4py import PETSc

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


@unittest.skipIf(MPI is None, 'current SF API needs mpi4py')
class TestSF(unittest.TestCase):
    def setUp(self):
        self.sf = PETSc.SF().create(PETSc.COMM_SELF)
        self.sf.setGraph(2, [0, 2], [[0, 1], [0, 0]])
        self.sf.setUp()

    def tearDown(self):
        self.sf.destroy()
        self.sf = None
        PETSc.garbage_cleanup()

    def testCollectives(self):
        root = np.array([10, 20], dtype=np.intc)
        leaf = np.full(3, -1, dtype=np.intc)
        self.sf.bcastBegin(MPI.INT, root, leaf, MPI.REPLACE)
        self.sf.bcastEnd(MPI.INT, root, leaf, MPI.REPLACE)
        np.testing.assert_array_equal(leaf, [20, -1, 10])

        root.fill(0)
        self.sf.reduceBegin(MPI.INT, leaf, root, MPI.SUM)
        self.sf.reduceEnd(MPI.INT, leaf, root, MPI.SUM)
        np.testing.assert_array_equal(root, [10, 20])

        multiroot = np.empty(2, dtype=np.intc)
        self.sf.gatherBegin(MPI.INT, leaf, multiroot)
        self.sf.gatherEnd(MPI.INT, leaf, multiroot)
        scattered = np.full(3, -1, dtype=np.intc)
        self.sf.scatterBegin(MPI.INT, multiroot, scattered)
        self.sf.scatterEnd(MPI.INT, multiroot, scattered)
        np.testing.assert_array_equal(scattered, leaf)

        root[:] = [100, 200]
        leaf[:] = [3, 0, 4]
        leafupdate = np.full(3, -1, dtype=np.intc)
        self.sf.fetchAndOpBegin(MPI.INT, root, leaf, leafupdate, MPI.SUM)
        self.sf.fetchAndOpEnd(MPI.INT, root, leaf, leafupdate, MPI.SUM)
        np.testing.assert_array_equal(root, [104, 203])
        np.testing.assert_array_equal(leafupdate, [200, -1, 100])

    def testRejectsNoncontiguousBuffer(self):
        root = np.arange(4, dtype=np.intc)[::2]
        leaf = np.empty(3, dtype=np.intc)
        with self.assertRaisesRegex(ValueError, 'rootdata must be contiguous'):
            self.sf.bcastBegin(MPI.INT, root, leaf, MPI.REPLACE)

    def testRejectsReadonlyOutput(self):
        root = np.empty(2, dtype=np.intc)
        leaf = np.empty(3, dtype=np.intc)
        leaf.flags.writeable = False
        with self.assertRaisesRegex(ValueError, 'leafdata must be writable'):
            self.sf.bcastBegin(MPI.INT, root, leaf, MPI.REPLACE)

    def testFetchAndOpRejectsNonarrayBuffers(self):
        root = np.array([10, 20], dtype=np.intc)
        leaf = np.array([3, 0, 4], dtype=np.intc)
        leafupdate = np.empty(3, dtype=np.intc)
        buffers = (root, leaf, leafupdate)
        for i, name in enumerate(('rootdata', 'leafdata', 'leafupdate')):
            for invalid in (buffers[i].tolist(), None):
                args = list(buffers)
                args[i] = invalid
                with self.subTest(buffer=name, type=type(invalid).__name__):
                    with self.assertRaises(TypeError):
                        self.sf.fetchAndOpBegin(MPI.INT, *args, MPI.SUM)
                    self.sf.fetchAndOpBegin(MPI.INT, *buffers, MPI.SUM)
                    try:
                        with self.assertRaises(TypeError):
                            self.sf.fetchAndOpEnd(MPI.INT, *args, MPI.SUM)
                    finally:
                        self.sf.fetchAndOpEnd(MPI.INT, *buffers, MPI.SUM)

    def testRejectsShortBuffers(self):
        root = np.empty(2, dtype=np.intc)
        leaf = np.empty(3, dtype=np.intc)
        with self.subTest(buffer='rootdata'):
            with self.assertRaisesRegex(ValueError, 'rootdata buffer is too small'):
                self.sf.bcastBegin(MPI.INT, root[:1], leaf, MPI.REPLACE)
        with self.subTest(buffer='leafdata'):
            with self.assertRaisesRegex(ValueError, 'leafdata buffer is too small'):
                self.sf.bcastBegin(MPI.INT, root, leaf[:2], MPI.REPLACE)
        with self.subTest(buffer='multirootdata'):
            with self.assertRaisesRegex(
                ValueError, 'multirootdata buffer is too small'
            ):
                self.sf.scatterBegin(MPI.INT, root[:1], leaf)


if __name__ == '__main__':
    unittest.main()
