from petsc4py import PETSc
import unittest, numpy
from sys import getrefcount
# --------------------------------------------------------------------

class Matrix(object):

    def __init__(self):
        pass

    def create(self, mat):
        pass

    def destroy(self, mat):
        pass

class ScaledIdentity(Matrix):

    s = 2.0

    def scale(self, mat, s):
        self.s *= s

    def shift(self, mat, s):
        self.s += s

    def mult(self, mat, x, y):
        x.copy(y)
        y.scale(self.s)

    def duplicate(self, mat, op):
        dmat = PETSc.Mat()
        dctx = ScaledIdentity()
        dmat.createPython(mat.getSizes(), dctx, comm=mat.getComm())
        if op == PETSc.Mat.DuplicateOption.COPY_VALUES:
          dctx.s = self.s
          dmat.setUp()
        return dmat

    def getDiagonal(self, mat, vd):
        vd.set(self.s)

    def productSetFromOptions(self, mat, producttype, A, B, C):
        return True

    def productSymbolic(self, mat, product, producttype, A, B, C):
        if producttype == 'AB':
            if mat is A: # product = identity * B
                product.setType(B.getType())
                product.setSizes(B.getSizes())
                product.setUp()
                product.assemble()
                B.copy(product)
            elif mat is B: # product = A * identity
                product.setType(A.getType())
                product.setSizes(A.getSizes())
                product.setUp()
                product.assemble()
                A.copy(product)
            else:
                raise RuntimeError('wrong configuration')
        elif producttype == 'AtB':
            if mat is A: # product = identity^T * B
                product.setType(B.getType())
                product.setSizes(B.getSizes())
                product.setUp()
                product.assemble()
                B.copy(product)
            elif mat is B: # product = A^T * identity
                tmp = PETSc.Mat()
                A.transpose(tmp)
                product.setType(tmp.getType())
                product.setSizes(tmp.getSizes())
                product.setUp()
                product.assemble()
                tmp.copy(product)
            else:
                raise RuntimeError('wrong configuration')
        elif producttype == 'ABt':
            if mat is A: # product = identity * B^T
                tmp = PETSc.Mat()
                B.transpose(tmp)
                product.setType(tmp.getType())
                product.setSizes(tmp.getSizes())
                product.setUp()
                product.assemble()
                tmp.copy(product)
            elif mat is B: # product = A * identity^T
                product.setType(A.getType())
                product.setSizes(A.getSizes())
                product.setUp()
                product.assemble()
                A.copy(product)
            else:
                raise RuntimeError('wrong configuration')
        elif producttype == 'PtAP':
            if mat is A: # product = P^T * identity * P
                self.tmp = PETSc.Mat()
                B.transposeMatMult(B, self.tmp)
                product.setType(self.tmp.getType())
                product.setSizes(self.tmp.getSizes())
                product.setUp()
                product.assemble()
                self.tmp.copy(product)
            elif mat is B: # product = identity^T * A * identity
                product.setType(A.getType())
                product.setSizes(A.getSizes())
                product.setUp()
                product.assemble()
                A.copy(product)
            else:
                raise RuntimeError('wrong configuration')
        elif producttype == 'RARt':
            if mat is A: # product = R * identity * R^t
                self.tmp = PETSc.Mat()
                B.matTransposeMult(B, self.tmp)
                product.setType(self.tmp.getType())
                product.setSizes(self.tmp.getSizes())
                product.setUp()
                product.assemble()
                self.tmp.copy(product)
            elif mat is B: # product = identity * A * identity^T
                product.setType(A.getType())
                product.setSizes(A.getSizes())
                product.setUp()
                product.assemble()
                A.copy(product)
            else:
                raise RuntimeError('wrong configuration')
        elif producttype == 'ABC':
            if mat is A: # product = identity * B * C
                self.tmp = PETSc.Mat()
                B.matMult(C, self.tmp)
                product.setType(self.tmp.getType())
                product.setSizes(self.tmp.getSizes())
                product.setUp()
                product.assemble()
                self.tmp.copy(product)
            elif mat is B: # product = A * identity * C
                self.tmp = PETSc.Mat()
                A.matMult(C, self.tmp)
                product.setType(self.tmp.getType())
                product.setSizes(self.tmp.getSizes())
                product.setUp()
                product.assemble()
                self.tmp.copy(product)
            elif mat is C: # product = A * B * identity
                self.tmp = PETSc.Mat()
                A.matMult(B, self.tmp)
                product.setType(self.tmp.getType())
                product.setSizes(self.tmp.getSizes())
                product.setUp()
                product.assemble()
                self.tmp.copy(product)
            else:
                raise RuntimeError('wrong configuration')
        else:
            raise RuntimeError('Product {} not implemented'.format(producttype))
        product.zeroEntries()

    def productNumeric(self, mat, product, producttype, A, B, C):
        if producttype == 'AB':
            if mat is A: # product = identity * B
                B.copy(product, structure=True)
            elif mat is B: # product = A * identity
                A.copy(product, structure=True)
            else:
                raise RuntimeError('wrong configuration')
            product.scale(self.s)
        elif producttype == 'AtB':
            if mat is A: # product = identity^T * B
                B.copy(product, structure=True)
            elif mat is B: # product = A^T * identity
                A.setTransposePrecursor(product)
                A.transpose(product)
            else:
                raise RuntimeError('wrong configuration')
            product.scale(self.s)
        elif producttype == 'ABt':
            if mat is A: # product = identity * B^T
                B.setTransposePrecursor(product)
                B.transpose(product)
            elif mat is B: # product = A * identity^T
                A.copy(product, structure=True)
            else:
                raise RuntimeError('wrong configuration')
            product.scale(self.s)
        elif producttype == 'PtAP':
            if mat is A: # product = P^T * identity * P
                B.transposeMatMult(B, self.tmp)
                self.tmp.copy(product, structure=True)
                product.scale(self.s)
            elif mat is B: # product = identity^T * A * identity
                A.copy(product, structure=True)
                product.scale(self.s**2)
            else:
                raise RuntimeError('wrong configuration')
        elif producttype == 'RARt':
            if mat is A: # product = R * identity * R^t
                B.matTransposeMult(B, self.tmp)
                self.tmp.copy(product, structure=True)
                product.scale(self.s)
            elif mat is B: # product = identity * A * identity^T
                A.copy(product, structure=True)
                product.scale(self.s**2)
            else:
                raise RuntimeError('wrong configuration')
        elif producttype == 'ABC':
            if mat is A: # product = identity * B * C
                B.matMult(C, self.tmp)
                self.tmp.copy(product, structure=True)
            elif mat is B: # product = A * identity * C
                A.matMult(C, self.tmp)
                self.tmp.copy(product, structure=True)
            elif mat is C: # product = A * B * identity
                A.matMult(B, self.tmp)
                self.tmp.copy(product, structure=True)
            else:
                raise RuntimeError('wrong configuration')
            product.scale(self.s)
        else:
            raise RuntimeError('Product {} not implemented'.format(producttype))

class Diagonal(Matrix):

    def create(self, mat):
        super(Diagonal,self).create(mat)
        mat.setUp()
        self.D = mat.createVecLeft()

    def destroy(self, mat):
        self.D.destroy()
        super(Diagonal,self).destroy(mat)

    def scale(self, mat, a):
        self.D.scale(a)

    def shift(self, mat, a):
        self.D.shift(a)

    def zeroEntries(self, mat):
        self.D.zeroEntries()

    def mult(self, mat, x, y):
        y.pointwiseMult(x, self.D)

    def duplicate(self, mat, op):
        dmat = PETSc.Mat()
        dctx = Diagonal()
        dmat.createPython(mat.getSizes(), dctx, comm=mat.getComm())
        dctx.D = self.D.duplicate()
        if op == PETSc.Mat.DuplicateOption.COPY_VALUES:
          self.D.copy(dctx.D)
          dmat.setUp()
        return dmat

    def getDiagonal(self, mat, vd):
        self.D.copy(vd)

    def setDiagonal(self, mat, vd, im):
        if isinstance (im, bool):
            addv = im
            if addv:
                self.D.axpy(1, vd)
            else:
                vd.copy(self.D)
        elif im == PETSc.InsertMode.INSERT_VALUES:
            vd.copy(self.D)
        elif im == PETSc.InsertMode.ADD_VALUES:
            self.D.axpy(1, vd)
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

    def _getCtx(self):
        return self.A.getPythonContext()

    def setUp(self):
        N = self.N = 13
        self.A = PETSc.Mat()
        if 0: # command line way
            self.A.create(self.COMM)
            self.A.setSizes([N,N])
            self.A.setType('python')
            OptDB = PETSc.Options(self.A)
            OptDB['mat_python_type'] = '%s.%s' % (self.PYMOD,self.PYCLS)
            self.A.setFromOptions()
            self.A.setUp()
            del OptDB['mat_python_type']
            self.assertTrue(self._getCtx() is not None)
        else: # python way
            context = globals()[self.PYCLS]()
            self.A.createPython([N,N], context, comm=self.COMM)
            self.A.setUp()
            self.assertTrue(self._getCtx() is context)
            self.assertEqual(getrefcount(context), 3)
            del context
            self.assertEqual(getrefcount(self._getCtx()), 2)

    def tearDown(self):
        ctx = self.A.getPythonContext()
        self.assertEqual(getrefcount(ctx), 3)
        self.A.destroy() # XXX
        self.A = None
        self.assertEqual(getrefcount(ctx), 2)
        #import gc,pprint; pprint.pprint(gc.get_referrers(ctx))

    def testBasic(self):
        ctx = self.A.getPythonContext()
        self.assertTrue(self._getCtx() is ctx)
        self.assertEqual(getrefcount(ctx), 3)

    def testZeroEntries(self):
        f = lambda : self.A.zeroEntries()
        self.assertRaises(Exception, f)

    def testMult(self):
        x, y = self.A.createVecs()
        f = lambda : self.A.mult(x, y)
        self.assertRaises(Exception, f)

    def testMultTranspose(self):
        x, y = self.A.createVecs()
        f = lambda : self.A.multTranspose(x, y)
        self.assertRaises(Exception, f)

    def testGetDiagonal(self):
        d = self.A.createVecLeft()
        f = lambda : self.A.getDiagonal(d)
        self.assertRaises(Exception, f)

    def testSetDiagonal(self):
        d = self.A.createVecLeft()
        f = lambda : self.A.setDiagonal(d)
        self.assertRaises(Exception, f)

    def testDiagonalScale(self):
        x, y = self.A.createVecs()
        f = lambda : self.A.diagonalScale(x, y)
        self.assertRaises(Exception, f)

    def testDuplicate(self):
        f1 = lambda : self.A.duplicate(x, True)
        f2 = lambda : self.A.duplicate(x, False)
        self.assertRaises(Exception, f1)
        self.assertRaises(Exception, f2)

    def testSetVecType(self):
        self.A.setVecType('mpi')
        self.assertTrue('mpi' == self.A.getVecType())

    def testH2Opus(self):
        if not PETSc.Sys.hasExternalPackage("h2opus"):
            return
        if self.A.getComm().Get_size() > 1:
            return
        h = PETSc.Mat()

        # need matrix vector and its transpose for norm estimation
        AA = self.A.getPythonContext()
        if not hasattr(AA,'mult'):
            return
        AA.multTranspose = AA.mult

        # without coordinates
        h.createH2OpusFromMat(self.A,leafsize=2)
        h.assemble()
        h.destroy()

        # with coordinates
        coords = numpy.linspace((1,2,3),(10,20,30),self.A.getSize()[0],dtype=PETSc.RealType)
        h.createH2OpusFromMat(self.A,coords,leafsize=2)
        h.assemble()

        # test API
        h.H2OpusOrthogonalize()
        h.H2OpusCompress(1.e-1)

        # Low-rank update
        U = PETSc.Mat()
        U.createDense([h.getSizes()[0],3],comm=h.getComm())
        U.setUp()
        U.setRandom()

        he = PETSc.Mat()
        h.convert('dense',he)
        he.axpy(1.0, U.matTransposeMult(U))

        h.H2OpusLowRankUpdate(U)
        self.assertTrue(he.equal(h))


        h.destroy()

        del AA.multTranspose


class TestScaledIdentity(TestMatrix):

    PYCLS = 'ScaledIdentity'

    def testMult(self):
        s = self._getCtx().s
        x, y = self.A.createVecs()
        x.setRandom()
        self.A.mult(x,y)
        self.assertTrue(y.equal(s*x))

    def testMultTransposeSymmKnown(self):
        s = self._getCtx().s
        x, y = self.A.createVecs()
        x.setRandom()
        self.A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.A.multTranspose(x,y)
        self.assertTrue(y.equal(s*x))
        self.A.setOption(PETSc.Mat.Option.SYMMETRIC, False)
        f = lambda : self.A.multTranspose(x, y)
        self.assertRaises(Exception, f)

    def testMultTransposeNewMeth(self):
        s = self._getCtx().s
        x, y = self.A.createVecs()
        x.setRandom()
        AA = self.A.getPythonContext()
        AA.multTranspose = AA.mult
        self.A.multTranspose(x,y)
        del AA.multTranspose
        self.assertTrue(y.equal(s*x))

    def testGetDiagonal(self):
        s = self._getCtx().s
        d = self.A.createVecLeft()
        o = d.duplicate()
        o.set(s)
        self.A.getDiagonal(d)
        self.assertTrue(o.equal(d))

    def testDuplicate(self):
        B = self.A.duplicate(False)
        self.assertTrue(B.getPythonContext().s == 2)
        B = self.A.duplicate(True)
        self.assertTrue(B.getPythonContext().s == self.A.getPythonContext().s)

    def testMatMat(self):
        s = self._getCtx().s
        R = PETSc.Random().create(self.COMM)
        R.setFromOptions()
        A = PETSc.Mat().create(self.COMM)
        A.setSizes(self.A.getSizes())
        A.setType(PETSc.Mat.Type.AIJ)
        A.setUp()
        A.setRandom(R)
        B = PETSc.Mat().create(self.COMM)
        B.setSizes(self.A.getSizes())
        B.setType(PETSc.Mat.Type.AIJ)
        B.setUp()
        B.setRandom(R)
        I = PETSc.Mat().create(self.COMM)
        I.setSizes(self.A.getSizes())
        I.setType(PETSc.Mat.Type.AIJ)
        I.setUp()
        I.assemble()
        I.shift(s)

        self.assertTrue(self.A.matMult(A).equal(I.matMult(A)))
        self.assertTrue(A.matMult(self.A).equal(A.matMult(I)))
        if self.A.getComm().Get_size() == 1:
            self.assertTrue(self.A.matTransposeMult(A).equal(I.matTransposeMult(A)))
            self.assertTrue(A.matTransposeMult(self.A).equal(A.matTransposeMult(I)))
        self.assertTrue(self.A.transposeMatMult(A).equal(I.transposeMatMult(A)))
        self.assertTrue(A.transposeMatMult(self.A).equal(A.transposeMatMult(I)))
        self.assertAlmostEqual((self.A.ptap(A) - I.ptap(A)).norm(), 0.0, places=5)
        self.assertAlmostEqual((A.ptap(self.A) - A.ptap(I)).norm(), 0.0, places=5)
        if self.A.getComm().Get_size() == 1:
            self.assertAlmostEqual((self.A.rart(A) - I.rart(A)).norm(), 0.0, places=5)
            self.assertAlmostEqual((A.rart(self.A) - A.rart(I)).norm(), 0.0, places=5)
        self.assertAlmostEqual((self.A.matMatMult(A,B)-I.matMatMult(A,B)).norm(), 0.0, places=5)
        self.assertAlmostEqual((A.matMatMult(self.A,B)-A.matMatMult(I,B)).norm(), 0.0, places=5)
        self.assertAlmostEqual((A.matMatMult(B,self.A)-A.matMatMult(B,I)).norm(), 0.0, places=5)

    def testShift(self):
        sold = self._getCtx().s
        self.A.shift(-0.5)
        s = self._getCtx().s
        self.assertTrue(s == sold - 0.5)

    def testScale(self):
        sold = self._getCtx().s
        self.A.scale(-0.5)
        s = self._getCtx().s
        self.assertTrue(s == sold * -0.5)

class TestDiagonal(TestMatrix):

    PYCLS = 'Diagonal'

    def setUp(self):
        super(TestDiagonal, self).setUp()
        D = self.A.createVecLeft()
        s, e = D.getOwnershipRange()
        for i in range(s, e):
            D[i] = i+1
        D.assemble()
        self.A.setDiagonal(D)

    def testZeroEntries(self):
        self.A.zeroEntries()
        D = self._getCtx().D
        self.assertEqual(D.norm(), 0)

    def testMult(self):
        x, y = self.A.createVecs()
        x.set(1)
        self.A.mult(x,y)
        self.assertTrue(y.equal(self._getCtx().D))

    def testMultTransposeSymmKnown(self):
        x, y = self.A.createVecs()
        x.set(1)
        self.A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        self.A.multTranspose(x,y)
        self.assertTrue(y.equal(self._getCtx().D))
        self.A.setOption(PETSc.Mat.Option.SYMMETRIC, False)
        f = lambda : self.A.multTranspose(x, y)
        self.assertRaises(Exception, f)

    def testMultTransposeNewMeth(self):
        x, y = self.A.createVecs()
        x.set(1)
        AA = self.A.getPythonContext()
        AA.multTranspose = AA.mult
        self.A.multTranspose(x,y)
        del AA.multTranspose
        self.assertTrue(y.equal(self._getCtx().D))

    def testDuplicate(self):
        B = self.A.duplicate(False)
        B = self.A.duplicate(True)
        self.assertTrue(B.getPythonContext().D.equal(self.A.getPythonContext().D))

    def testGetDiagonal(self):
        d = self.A.createVecLeft()
        self.A.getDiagonal(d)
        self.assertTrue(d.equal(self._getCtx().D))

    def testSetDiagonal(self):
        d = self.A.createVecLeft()
        d.setRandom()
        self.A.setDiagonal(d)
        self.assertTrue(d.equal(self._getCtx().D))

    def testDiagonalScale(self):
        x, y = self.A.createVecs()
        x.set(2)
        y.set(3)
        old = self._getCtx().D.copy()
        self.A.diagonalScale(x, y)
        D = self._getCtx().D
        self.assertTrue(D.equal(old*6))

    def testCreateTranspose(self):
        A = self.A
        A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        AT = PETSc.Mat().createTranspose(A)
        x, y = A.createVecs()
        xt, yt = AT.createVecs()
        #
        y.setRandom()
        A.multTranspose(y, x)
        y.copy(xt)
        AT.mult(xt, yt)
        self.assertTrue(yt.equal(x))
        #
        x.setRandom()
        A.mult(x, y)
        x.copy(yt)
        AT.multTranspose(yt, xt)
        self.assertTrue(xt.equal(y))
        del A

    def testConvert(self):
        self.assertTrue(self.A.convert(PETSc.Mat.Type.AIJ,PETSc.Mat()).equal(self.A))
        self.assertTrue(self.A.convert(PETSc.Mat.Type.BAIJ,PETSc.Mat()).equal(self.A))
        self.assertTrue(self.A.convert(PETSc.Mat.Type.SBAIJ,PETSc.Mat()).equal(self.A))
        self.assertTrue(self.A.convert(PETSc.Mat.Type.DENSE,PETSc.Mat()).equal(self.A))

    def testShift(self):
        old = self._getCtx().D.copy()
        self.A.shift(-0.5)
        D = self._getCtx().D
        self.assertTrue(D.equal(old-0.5))

    def testScale(self):
        old = self._getCtx().D.copy()
        self.A.scale(-0.5)
        D = self._getCtx().D
        self.assertTrue(D.equal(-0.5*old))


# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
