# --------------------------------------------------------------------

from petsc4py import PETSc
import unittest
from sys import getrefcount

# --------------------------------------------------------------------

class BaseMyPC(object):
    def setup(self, pc):
        pass
    def reset(self, pc):
        pass
    def apply(self, pc, x, y):
        raise NotImplementedError
    def applyT(self, pc, x, y):
        self.apply(pc, x, y)
    def applyS(self, pc, x, y):
        self.apply(pc, x, y)
    def applySL(self, pc, x, y):
        self.applyS(pc, x, y)
    def applySR(self, pc, x, y):
        self.applyS(pc, x, y)
    def applyRich(self, pc, x, y, w, tols):
        self.apply(pc, x, y)
    def applyM(self, pc, x, y):
        raise NotImplementedError

class MyPCNone(BaseMyPC):
    def apply(self, pc, x, y):
        x.copy(y)
    def applyM(self, pc, x, y):
        x.copy(y)

class MyPCJacobi(BaseMyPC):
    def setup(self, pc):
        A, P = pc.getOperators()
        self.diag = P.getDiagonal()
        self.diag.reciprocal()
    def reset(self, pc):
        self.diag.destroy()
        del self.diag
    def apply(self, pc, x, y):
        y.pointwiseMult(self.diag, x)
    def applyS(self, pc, x, y):
        self.diag.copy(y)
        y.sqrtabs()
        y.pointwiseMult(y, x)
    def applyM(self, pc, x, y):
        x.copy(y)
        y.diagonalScale(L=self.diag)

class PC_PYTHON_CLASS(object):

    def __init__(self):
        self.impl = None
        self.log = {}
    def _log(self, method, *args):
        self.log.setdefault(method, 0)
        self.log[method] += 1
    def create(self, pc):
        self._log('create', pc)
    def destroy(self, pc):
        self._log('destroy')
        self.impl = None
    def reset(self, pc):
        self._log('reset', pc)
    def view(self, pc, vw):
        self._log('view', pc, vw)
        assert isinstance(pc, PETSc.PC)
        assert isinstance(vw, PETSc.Viewer)
        pass
    def setFromOptions(self, pc):
        self._log('setFromOptions', pc)
        assert isinstance(pc, PETSc.PC)
        OptDB = PETSc.Options(pc)
        impl =  OptDB.getString('impl','MyPCNone')
        klass = globals()[impl]
        self.impl = klass()
    def setUp(self, pc):
        self._log('setUp', pc)
        assert isinstance(pc, PETSc.PC)
        self.impl.setup(pc)
    def preSolve(self, pc, ksp, b, x):
        self._log('preSolve', pc, ksp, b, x)
    def postSolve(self, pc, ksp, b, x):
        self._log('postSolve', pc, ksp, b, x)
    def apply(self, pc, x, y):
        self._log('apply', pc, x, y)
        assert isinstance(pc, PETSc.PC)
        assert isinstance(x,  PETSc.Vec)
        assert isinstance(y,  PETSc.Vec)
        self.impl.apply(pc, x, y)
    def applySymmetricLeft(self, pc, x, y):
        self._log('applySymmetricLeft', pc, x, y)
        assert isinstance(pc, PETSc.PC)
        assert isinstance(x,  PETSc.Vec)
        assert isinstance(y,  PETSc.Vec)
        self.impl.applySL(pc, x, y)
    def applySymmetricRight(self, pc, x, y):
        self._log('applySymmetricRight', pc, x, y)
        assert isinstance(pc, PETSc.PC)
        assert isinstance(x,  PETSc.Vec)
        assert isinstance(y,  PETSc.Vec)
        self.impl.applySR(pc, x, y)
    def applyTranspose(self, pc, x, y):
        self._log('applyTranspose', pc, x, y)
        assert isinstance(pc, PETSc.PC)
        assert isinstance(x,  PETSc.Vec)
        assert isinstance(y,  PETSc.Vec)
        self.impl.applyT(pc, x, y)
    def matApply(self, pc, x, y):
        self._log('matApply', pc, x, y)
        assert isinstance(pc, PETSc.PC)
        assert isinstance(x,  PETSc.Mat)
        assert isinstance(y,  PETSc.Mat)
        self.impl.applyM(pc, x, y)
    def applyRichardson(self, pc, x, y, w, tols):
        self._log('applyRichardson', pc, x, y, w, tols)
        assert isinstance(pc, PETSc.PC)
        assert isinstance(x,  PETSc.Vec)
        assert isinstance(y,  PETSc.Vec)
        assert isinstance(w,  PETSc.Vec)
        assert isinstance(tols,  tuple)
        assert len(tols) == 4
        self.impl.applyRich(pc, x, y, w, tols)


class TestPCPYTHON(unittest.TestCase):

    PC_TYPE = PETSc.PC.Type.PYTHON
    PC_PREFIX = 'test-'

    def setUp(self):
        pc = self.pc = PETSc.PC()
        pc.create(PETSc.COMM_SELF)
        pc.setType(self.PC_TYPE)
        module = __name__
        factory = 'PC_PYTHON_CLASS'
        self.pc.prefix = self.PC_PREFIX
        OptDB = PETSc.Options(self.pc)
        assert OptDB.prefix == self.pc.prefix
        OptDB['pc_python_type'] = '%s.%s' % (module, factory)
        self.pc.setFromOptions()
        del OptDB['pc_python_type']
        assert self._getCtx().log['create'] == 1
        assert self._getCtx().log['setFromOptions'] == 1
        ctx = self._getCtx()
        self.assertEqual(getrefcount(ctx), 3)

    def testGetType(self):
        ctx = self.pc.getPythonContext()
        pytype = "{0}.{1}".format(ctx.__module__, type(ctx).__name__)
        self.assertTrue(self.pc.getPythonType() == pytype)

    def tearDown(self):
        ctx = self._getCtx()
        self.pc.destroy() # XXX
        self.pc = None
        assert ctx.log['destroy'] == 1
        self.assertEqual(getrefcount(ctx), 2)

    def _prepare(self):
        A = PETSc.Mat().createAIJ([3,3], comm=PETSc.COMM_SELF)
        A.setUp()
        A.assemble()
        A.shift(10)
        x, y = A.createVecs()
        x.setRandom()
        self.pc.setOperators(A, A)
        X = PETSc.Mat().createDense([3,5], comm=PETSc.COMM_SELF).setUp()
        X.assemble()
        Y = PETSc.Mat().createDense([3,5], comm=PETSc.COMM_SELF).setUp()
        Y.assemble()
        assert (A,A) == self.pc.getOperators()
        return A, x, y, X, Y

    def _getCtx(self):
        return self.pc.getPythonContext()

    def _applyMeth(self, meth):
        A, x, y, X, Y = self._prepare()
        if meth == 'matApply':
            getattr(self.pc, meth)(X,Y)
            x.copy(y)
        else:
            getattr(self.pc, meth)(x,y)
            X.copy(Y)
        if 'reset' not in self._getCtx().log:
            assert self._getCtx().log['setUp'] == 1
            assert self._getCtx().log[meth] == 1
        else:
            nreset = self._getCtx().log['reset']
            nsetup = self._getCtx().log['setUp']
            nmeth  = self._getCtx().log[meth]
            assert (nreset == nsetup)
            assert (nreset == nmeth)
        if isinstance(self._getCtx().impl, MyPCNone):
            self.assertTrue(y.equal(x))
            self.assertTrue(Y.equal(X))
    def testApply(self):
        self._applyMeth('apply')
    def testApplySymmetricLeft(self):
        self._applyMeth('applySymmetricLeft')
    def testApplySymmetricRight(self):
        self._applyMeth('applySymmetricRight')
    def testApplyTranspose(self):
        self._applyMeth('applyTranspose')
    def testApplyMat(self):
        self._applyMeth('matApply')
    ## def testApplyRichardson(self):
    ##     x, y = self._prepare()
    ##     w = x.duplicate()
    ##     tols = 0,0,0,0
    ##     self.pc.applyRichardson(x,y,w,tols)
    ##     assert self._getCtx().log['setUp'] == 1
    ##     assert self._getCtx().log['applyRichardson'] == 1

    ## def testView(self):
    ##     vw = PETSc.ViewerString(100, self.pc.comm)
    ##     self.pc.view(vw)
    ##     s = vw.getString()
    ##     assert 'python' in s
    ##     module = __name__
    ##     factory = 'self._getCtx()'
    ##     assert '.'.join([module, factory]) in s

    def testResetAndApply(self):
        self.pc.reset()
        self.testApply()
        self.pc.reset()
        self.testApply()
        self.pc.reset()

    def testKSPSolve(self):
        A, x, y, _, _ = self._prepare()
        ksp = PETSc.KSP().create(self.pc.comm)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        assert self.pc.getRefCount() == 1
        ksp.setPC(self.pc)
        assert self.pc.getRefCount() == 2
        # normal ksp solve, twice
        ksp.solve(x,y)
        assert self._getCtx().log['setUp'    ] == 1
        assert self._getCtx().log['apply'    ] == 1
        assert self._getCtx().log['preSolve' ] == 1
        assert self._getCtx().log['postSolve'] == 1
        ksp.solve(x,y)
        assert self._getCtx().log['setUp'    ] == 1
        assert self._getCtx().log['apply'    ] == 2
        assert self._getCtx().log['preSolve' ] == 2
        assert self._getCtx().log['postSolve'] == 2
        # transpose ksp solve, twice
        ksp.solveTranspose(x,y)
        assert self._getCtx().log['setUp'         ] == 1
        assert self._getCtx().log['applyTranspose'] == 1
        ksp.solveTranspose(x,y)
        assert self._getCtx().log['setUp'         ] == 1
        assert self._getCtx().log['applyTranspose'] == 2
        del ksp # ksp.destroy()
        assert self.pc.getRefCount() == 1

    def testGetSetContext(self):
        ctx = self.pc.getPythonContext()
        self.pc.setPythonContext(ctx)
        self.assertEqual(getrefcount(ctx), 3)
        del ctx


class TestPCPYTHON2(TestPCPYTHON):
    def setUp(self):
        OptDB = PETSc.Options(self.PC_PREFIX)
        OptDB['impl'] = 'MyPCJacobi'
        super(TestPCPYTHON2, self).setUp()
        clsname = type(self._getCtx().impl).__name__
        assert clsname == OptDB['impl']
        del OptDB['impl']

class TestPCPYTHON3(TestPCPYTHON):
    def setUp(self):
        pc = self.pc = PETSc.PC()
        ctx = PC_PYTHON_CLASS()
        pc.createPython(ctx, comm=PETSc.COMM_SELF)
        self.pc.prefix = self.PC_PREFIX
        self.pc.setFromOptions()
        assert self._getCtx().log['create'] == 1
        assert self._getCtx().log['setFromOptions'] == 1

class TestPCPYTHON4(TestPCPYTHON3):
    def setUp(self):
        OptDB = PETSc.Options(self.PC_PREFIX)
        OptDB['impl'] = 'MyPCJacobi'
        super(TestPCPYTHON4, self).setUp()
        clsname = type(self._getCtx().impl).__name__
        assert clsname == OptDB['impl']
        del OptDB['impl']

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
