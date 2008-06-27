from petsc4py import PETSc
import unittest
from sys import getrefcount
import gc

# --------------------------------------------------------------------

class Matrix(object):

    SELF = None

    def __init__(self):
        pass

    def create(self, mat):
        Matrix.SELF = self

    def destroy(self):
        Matrix.SELF = None

class Identity(Matrix):

    def mult(self, mat, x, y):
        x.copy(y)

    def getDiagonal(self, mat, d):
        d.set(1)

class Diagonal(Matrix):

    def create(self, mat):
        super(Diagonal,self).create(mat)
        self.D = mat.getVecs('l')

    def destroy(self):
        self.D.destroy()
        super(Diagonal,self).destroy()

    def scale(self, mat, a):
        self.D.scale(a)

    def shift(self, mat, a):
        self.D.shift(a)

    def zeroEntries(self, mat):
        self.D.zeroEntries()

    def mult(self, mat, x, y):
        y.pointwiseMult(x, self.D)

    def getDiagonal(self, mat, d):
        self.D.copy(d)

    def setDiagonal(self, mat, d, im):
        if im == PETSc.InsertMode.INSERT_VALUES:
            d.copy(self.D)
        elif im == PETSc.InsertMode.ADD_VALUES:
            self.D.axpy(1, d)
        else:
            raise ValueError('wrong InsertMode %d'% im)

    def diagonalScale(self, mat, vl, vr):
        if vl: self.D.pointwiseMult(self.D, vl)
        if vr: self.D.pointwiseMult(self.D, vr)

# --------------------------------------------------------------------

class TestMatrix(unittest.TestCase):

    COMM = PETSc.COMM_WORLD
    PYMOD = __name__
    PYCLS = 'Matrix'

    def setUp(self):
        N = self.N = 10
        A = self.mat = PETSc.Mat()
        if 0: # command line way
            A.create(self.COMM)
            A.setSizes([N,N])
            A.setType('python')
            OptDB = PETSc.Options(A)
            OptDB['mat_python'] = ','.join([self.PYMOD,self.PYCLS])
            A.setFromOptions()
            A.setUp()
            del OptDB['mat_python']
            self.assertTrue(Matrix.SELF is not None)
        else: # python way
            context = globals()[self.PYCLS]()
            A.createPython([N,N], context, comm=self.COMM)
            self.assertTrue(Matrix.SELF is context)
            self.assertEqual(getrefcount(context), 4)
            del context
            self.assertEqual(getrefcount(Matrix.SELF), 3)

    def tearDown(self):
        ctx = self.mat.getPythonContext()
        self.assertEqual(getrefcount(ctx), 4)
        self.assertTrue(Matrix.SELF is ctx)
        self.mat.destroy() # XXX
        self.mat = None
        self.assertTrue(Matrix.SELF is None)
        self.assertEqual(getrefcount(ctx), 2)

    def testBasic(self):
        ctx = self.mat.getPythonContext()
        self.assertTrue(Matrix.SELF is ctx)
        self.assertEqual(getrefcount(ctx), 4)

    def testZeroEntries(self):
        A = self.mat
        f = lambda : A.zeroEntries()
        self.assertRaises(Exception, f)

    def testMult(self):
        A = self.mat
        x, y = A.getVecs()
        f = lambda : A.mult(x, y)
        self.assertRaises(Exception, f)

    def testMultTranspose(self):
        A = self.mat
        x, y = A.getVecs()
        f = lambda : A.multTranspose(x, y)
        self.assertRaises(Exception, f)

    def testGetDiagonal(self):
        A = self.mat
        d = A.getVecs('l')
        f = lambda : A.getDiagonal(d)
        self.assertRaises(Exception, f)

    def testSetDiagonal(self):
        A = self.mat
        d = A.getVecs('l')
        f = lambda : A.setDiagonal(d)
        self.assertRaises(Exception, f)

    def testDiagonalScale(self):
        A = self.mat
        x, y = A.getVecs()
        f = lambda : A.diagonalScale(x, y)
        self.assertRaises(Exception, f)

class TestIdentity(TestMatrix):

    PYCLS = 'Identity'

    def testMult(self):
        A = self.mat
        x, y = A.getVecs()
        x.setRandom()
        A.mult(x,y)
        self.assertTrue(y.equal(x))

    def testMultTransposeSymmKnown(self):
        A = self.mat
        x, y = A.getVecs()
        x.setRandom()
        A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        A.multTranspose(x,y)
        self.assertTrue(y.equal(x))
        A.setOption(PETSc.Mat.Option.SYMMETRIC, False)
        f = lambda : A.multTranspose(x, y)
        self.assertRaises(Exception, f)

    def testMultTransposeNewMeth(self):
        A = self.mat
        x, y = A.getVecs()
        x.setRandom()
        AA = A.getPythonContext()
        AA.multTranspose = AA.mult
        A.multTranspose(x,y)
        del AA.multTranspose
        self.assertTrue(y.equal(x))

    def testGetDiagonal(self):
        A = self.mat
        d = A.getVecs('l')
        o = d.duplicate()
        o.set(1)
        A.getDiagonal(d)
        self.assertTrue(o.equal(d))


class TestDiagonal(TestMatrix):

    PYCLS = 'Diagonal'

    def setUp(self):
        super(TestDiagonal, self).setUp()
        A = self.mat
        D = A.getVecs('l')
        s, e = D.getOwnershipRange()
        for i in range(s, e):
            D[i] = i+1
        D.assemble()
        A.setDiagonal(D)


    def testZeroEntries(self):
        A = self.mat
        A.zeroEntries()
        D = Matrix.SELF.D
        self.assertEqual(D.norm(), 0)

    def testMult(self):
        A = self.mat
        x, y = A.getVecs()
        x.set(1)
        A.mult(x,y)
        self.assertTrue(y.equal(Matrix.SELF.D))

    def testMultTransposeSymmKnown(self):
        A = self.mat
        x, y = A.getVecs()
        x.set(1)
        A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        A.multTranspose(x,y)
        self.assertTrue(y.equal(Matrix.SELF.D))
        A.setOption(PETSc.Mat.Option.SYMMETRIC, False)
        f = lambda : A.multTranspose(x, y)
        self.assertRaises(Exception, f)

    def testMultTransposeNewMeth(self):
        A = self.mat
        x, y = A.getVecs()
        x.set(1)
        AA = A.getPythonContext()
        AA.multTranspose = AA.mult
        A.multTranspose(x,y)
        del AA.multTranspose
        self.assertTrue(y.equal(Matrix.SELF.D))

    def testGetDiagonal(self):
        A = self.mat
        d = A.getVecs('l')
        A.getDiagonal(d)
        self.assertTrue(d.equal(Matrix.SELF.D))

    def testSetDiagonal(self):
        A = self.mat
        d = A.getVecs('l')
        d.setRandom()
        A.setDiagonal(d)
        self.assertTrue(d.equal(Matrix.SELF.D))

    def testDiagonalScale(self):
        A = self.mat
        x, y = A.getVecs()
        x.set(2)
        y.set(3)
        old = Matrix.SELF.D.copy()
        A.diagonalScale(x, y)
        D = Matrix.SELF.D
        self.assertTrue(D.equal(old*6))

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
