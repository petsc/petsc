# --------------------------------------------------------------------

from petsc4py import PETSc
import unittest
from sys import getrefcount

# --------------------------------------------------------------------

class MySNES(object):

    def __init__(self):
        self.trace = False
        self.call_log = {}

    def _log(self, method, *args):
        self.call_log.setdefault(method, 0)
        self.call_log[method] += 1
        if not self.trace: return
        clsname = self.__class__.__name__
        pargs = []
        for a in args:
            pargs.append(a)
            if isinstance(a, PETSc.Object):
                pargs[-1] = type(a).__name__
        pargs = tuple(pargs)
        print ('%-20s' % ('%s.%s%s'% (clsname, method, pargs)))

    def create(self,*args):
        self._log('create', *args)

    def destroy(self,*args):
        self._log('destroy', *args)
        if not self.trace: return
        for k, v in self.call_log.items():
            print ('%-20s %2d' % (k, v))

    def view(self, snes, viewer):
        self._log('view', snes, viewer)

    def setFromOptions(self, snes):
        OptDB = PETSc.Options(snes)
        self.trace = OptDB.getBool('trace',self.trace)
        self._log('setFromOptions',snes)

    def setUp(self, snes):
        self._log('setUp', snes)

    def reset(self, snes):
        self._log('reset', snes)

    #def preSolve(self, snes):
    #    self._log('preSolve', snes)
    #
    #def postSolve(self, snes):
    #    self._log('postSolve', snes)

    def preStep(self, snes):
        self._log('preStep', snes)

    def postStep(self, snes):
        self._log('postStep', snes)

    #def computeFunction(self, snes, x, F):
    #    self._log('computeFunction', snes, x, F)
    #    snes.computeFunction(x, F)
    #
    #def computeJacobian(self, snes, x, A, B):
    #    self._log('computeJacobian', snes, x, A, B)
    #    flag = snes.computeJacobian(x, A, B)
    #    return flag
    #
    #def linearSolve(self, snes, b, x):
    #    self._log('linearSolve', snes, b, x)
    #    snes.ksp.solve(b,x)
    #    ## return False # not succeed
    #    if snes.ksp.getConvergedReason() < 0:
    #        return False # not succeed
    #    return True # succeed
    #
    #def lineSearch(self, snes, x, y, F):
    #    self._log('lineSearch', snes, x, y, F)
    #    x.axpy(-1,y)
    #    snes.computeFunction(x, F)
    #    ## return False # not succeed
    #    return True # succeed


from test_snes import BaseTestSNES

class TestSNESPython(BaseTestSNES, unittest.TestCase):

    SNES_TYPE = PETSc.SNES.Type.PYTHON

    def setUp(self):
        super(TestSNESPython, self).setUp()
        self.snes.setPythonContext(MySNES())

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

# --------------------------------------------------------------------
