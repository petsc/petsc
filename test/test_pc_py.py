# --------------------------------------------------------------------

from petsc4py import PETSc
import unittest
from sys import getrefcount

# --------------------------------------------------------------------

class MyPCBase(object):
    def setup(self, pc):
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

class MyPCNone(MyPCBase):
    def apply(self, pc, x, y):
        x.copy(y)

class MyPCJacobi(MyPCBase):
    def setup(self, pc):
        A, P, ms = pc.getOperators()
        self.diag = P.getDiagonal()
        self.diag.reciprocal()
    def apply(self, pc, x, y):
        y.pointwiseMult(self.diag, x)
    def applyS(self, pc, x, y):
        self.diag.copy(y)
        y.sqrt()
        y.pointwiseMult(y, x)

PC_PYTHON = None

class PC_PYTHON_CLASS(object):

    def __init__(self):
        global PC_PYTHON
        PC_PYTHON = self
        self.impl = None
        self.log = {}
    def _log(self, method, *args):
        self.log.setdefault(method, 0)
        self.log[method] += 1
    def create(self, pc):
        self._log('create', pc)
    def destroy(self):
        self._log('destroy')
        self.impl = None
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
        OptDB['pc_python'] = '%s,%s' % (module, factory)
        self.pc.setFromOptions()
        del OptDB['pc_python']
        assert PC_PYTHON.log['create'] == 1
        assert PC_PYTHON.log['setFromOptions'] == 1
    def tearDown(self):
        self.pc.destroy() # XXX
        self.pc = None
        global PC_PYTHON
        assert PC_PYTHON.log['destroy'] == 1
        PC_PYTHON = None

    def _prepare(self):
        A = PETSc.Mat().createAIJ([3,3], comm=PETSc.COMM_SELF)
        A.setUp()
        A.assemble()
        A.shift(10)
        x, y = A.getVecs()
        x.setRandom()
        self.pc.setOperators(A, A, True)
        assert (A,A) == self.pc.getOperators()[:2]
        return A, x, y

    def _applyMeth(self, meth):
        A, x, y = self._prepare()
        getattr(self.pc, meth)(x,y)
        assert PC_PYTHON.log['setUp'] == 1
        assert PC_PYTHON.log[meth] == 1
        if isinstance(PC_PYTHON.impl, MyPCNone):
            self.assertTrue(y.equal(x))
    def testApply(self):
        self._applyMeth('apply')
    def testApplySymmetricLeft(self):
        self._applyMeth('applySymmetricLeft')
    def testApplySymmetricRight(self):
        self._applyMeth('applySymmetricRight')
    def testApplyTranspose(self):
        self._applyMeth('applyTranspose')
    ## def testApplyRichardson(self):
    ##     x, y = self._prepare()
    ##     w = x.duplicate()
    ##     tols = 0,0,0,0
    ##     self.pc.applyRichardson(x,y,w,tols)
    ##     assert PC_PYTHON.log['setUp'] == 1
    ##     assert PC_PYTHON.log['applyRichardson'] == 1

    ## def testView(self):
    ##     vw = PETSc.ViewerString(100, self.pc.comm)
    ##     self.pc.view(vw)
    ##     s = vw.getString()
    ##     assert 'python' in s
    ##     module = __name__
    ##     factory = 'PC_PYTHON'
    ##     assert '.'.join([module, factory]) in s

    def testKSPSolve(self):
        A, x, y = self._prepare()
        ksp = PETSc.KSP().create(self.pc.comm)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        assert self.pc.getRefCount() == 1
        ksp.setPC(self.pc)
        assert self.pc.getRefCount() == 2
        # normal ksp solve, twice
        ksp.solve(x,y)
        assert PC_PYTHON.log['setUp'    ] == 1
        assert PC_PYTHON.log['apply'    ] == 1
        assert PC_PYTHON.log['preSolve' ] == 1
        assert PC_PYTHON.log['postSolve'] == 1
        ksp.solve(x,y)
        assert PC_PYTHON.log['setUp'    ] == 1
        assert PC_PYTHON.log['apply'    ] == 2
        assert PC_PYTHON.log['preSolve' ] == 2
        assert PC_PYTHON.log['postSolve'] == 2
        # transpose ksp solve, twice
        ksp.solveTranspose(x,y)
        assert PC_PYTHON.log['setUp'         ] == 1
        assert PC_PYTHON.log['applyTranspose'] == 1
        ksp.solveTranspose(x,y)
        assert PC_PYTHON.log['setUp'         ] == 1
        assert PC_PYTHON.log['applyTranspose'] == 2
        del ksp # ksp.destroy()
        assert self.pc.getRefCount() == 1

    def testGetSetContext(self):
        ctx = self.pc.getPythonContext()
        self.pc.setPythonContext(ctx)
        del ctx


class TestPCPYTHON2(TestPCPYTHON):
    def setUp(self):
        OptDB = PETSc.Options(self.PC_PREFIX)
        OptDB['impl'] = 'MyPCJacobi'
        super(TestPCPYTHON2, self).setUp()
        clsname = type(PC_PYTHON.impl).__name__
        assert clsname == OptDB['impl']
        del OptDB['impl']

class TestPCPYTHON3(TestPCPYTHON):
    def setUp(self):
        pc = self.pc = PETSc.PC()
        ctx = PC_PYTHON_CLASS()
        pc.createPython(ctx, comm=PETSc.COMM_SELF)
        self.pc.prefix = self.PC_PREFIX
        self.pc.setFromOptions()
        assert PC_PYTHON.log['create'] == 1
        assert PC_PYTHON.log['setFromOptions'] == 1

class TestPCPYTHON4(TestPCPYTHON3):
    def setUp(self):
        OptDB = PETSc.Options(self.PC_PREFIX)
        OptDB['impl'] = 'MyPCJacobi'
        super(TestPCPYTHON4, self).setUp()
        clsname = type(PC_PYTHON.impl).__name__
        assert clsname == OptDB['impl']
        del OptDB['impl']


# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
