from petsc4py import PETSc
import unittest
import random

# --------------------------------------------------------------------

class BaseTestIS(object):

    TYPE = None

    def tearDown(self):
        self.iset = None
        PETSc.garbage_cleanup()

    def testGetType(self):
        istype = self.iset.getType()
        self.assertEqual(istype, self.TYPE)

    def testGetSize(self):
        lsize = self.iset.getLocalSize()
        gsize = self.iset.getSize()
        self.assertTrue(lsize <= gsize)

    def testDuplicate(self):
        iset = self.iset.duplicate()
        self.assertTrue(self.iset.equal(iset))
        del iset

    def testCopy(self):
        iset = self.iset.copy()
        self.assertTrue(self.iset.equal(iset))
        del iset

    def testEqual(self):
        self.assertTrue(self.iset.equal(self.iset))
        iset = self.iset.duplicate()
        self.assertTrue(self.iset.equal(iset))
        del iset

    def testSort(self):
        self.iset.sort()
        self.assertTrue(self.iset.isSorted())

    def testDifference(self):
        iset = self.iset.difference(self.iset)
        self.assertEqual(iset.getLocalSize(), 0)
        del iset

    def testComplement(self):
        self.iset.sort()
        nmin = self.iset.getIndices().min()
        nmax = self.iset.getIndices().max()
        iset = self.iset.complement(nmin, nmax+1)
        iset.complement(nmin, nmax+1)
        del iset

    def testSum(self):
        if self.iset.getComm().getSize() > 1:
            return
        self.iset.sort()
        iset = self.iset.duplicate()
        iset.sum(self.iset)
        self.assertTrue(self.iset.equal(iset))
        del iset

    def testExpand(self):
        iset = self.iset.expand(self.iset)
        if self.iset.type == iset.type:
            self.assertTrue(self.iset.equal(iset))
        del iset

    def testRenumber(self):
        (n1,is1) = self.iset.renumber()
        (n2,is2) = self.iset.renumber(self.iset)
        del is1
        del is2

    def testProperties(self):
        proplist = ['sizes', 'size', 'local_size', 'indices',
                    'permutation', 'identity', 'sorted']
        for prop in proplist:
            self.assertTrue(hasattr(self.iset, prop))

    def testArray(self):
        import numpy
        refs = self.iset.getRefCount()
        arr1 = numpy.asarray(self.iset)
        self.assertEqual(self.iset.getRefCount(), refs+1)
        arr2 = self.iset.array
        self.assertEqual(self.iset.getRefCount(), refs+2)
        self.assertTrue((arr1 == arr2).all())
        del arr2
        self.assertEqual(self.iset.getRefCount(), refs+1)
        del arr1
        self.assertEqual(self.iset.getRefCount(), refs)


# --------------------------------------------------------------------

class TestISGeneral(BaseTestIS, unittest.TestCase):

    TYPE = PETSc.IS.Type.GENERAL

    def setUp(self):
        self.idx = list(range(10))
        random.shuffle(self.idx)
        self.iset = PETSc.IS().createGeneral(self.idx)

    def testGetIndices(self):
        idx = self.iset.getIndices()
        self.assertEqual(self.idx, list(idx))


class TestISStride(BaseTestIS, unittest.TestCase):

    TYPE = PETSc.IS.Type.STRIDE

    def setUp(self):
        self.info = (10, 7, 3)
        size, start, step = self.info
        self.iset = PETSc.IS().createStride(size, start, step)

    def testGetIndices(self):
        size, start, step = self.info
        indices = [start+i*step for i in range(size)]
        self.assertEqual(list(self.iset.getIndices()), indices)

    def testToGeneral(self):
        self.iset.toGeneral()
        self.assertEqual(self.iset.getType(), PETSc.IS.Type.GENERAL)


class TestISBlock(BaseTestIS, unittest.TestCase):

    TYPE = PETSc.IS.Type.BLOCK

    def setUp(self):
        self.bsize = 3
        self.index = list(range(0,10,2))
        random.shuffle(self.index)
        self.iset = PETSc.IS().createBlock(self.bsize, self.index)
        self.assertEqual(self.iset.getType(), PETSc.IS.Type.BLOCK)

    def testGetSize(self):
        lsize = self.iset.getLocalSize()
        self.assertEqual(lsize/self.bsize, len(self.index))

    def testGetBlockSize(self):
        bs = self.iset.getBlockSize()
        self.assertEqual(bs, self.bsize)

    def testGetBlockIndices(self):
        index = list(self.iset.getBlockIndices())
        self.assertEqual(index, self.index)

    def testGetIndices(self):
        bs = self.bsize
        idx = []
        for i in self.iset.getBlockIndices():
            for j in range(bs):
                idx.append(i*bs+j)
        index = list(self.iset.getIndices())
        #self.assertEqual(index, idx)


# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
