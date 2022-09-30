import unittest
from petsc4py import PETSc
from sys import getrefcount

# --------------------------------------------------------------------
class Objective:
    def __call__(self, tao, x):
        return (x[0] - 1.0)**2 + (x[1] - 2.0)**2

class Gradient:
    def __call__(self, tao, x, g):
        g[0] = 2.0*(x[0] - 1.0)
        g[1] = 2.0*(x[1] - 2.0)
        g.assemble()

class MyTao:
    def __init__(self):
        self.log = {}

    def _log(self, method):
        self.log.setdefault(method, 0)
        self.log[method] += 1

    def create(self, tao):
        self._log('create')
        self.testvec = PETSc.Vec()

    def destroy(self, tao):
        self._log('destroy')
        self.testvec.destroy()

    def setFromOptions(self, tao):
        self._log('setFromOptions')

    def setUp(self, tao):
        self._log('setUp')
        self.testvec = tao.getSolution().duplicate()

    def solve(self, tao):
        self._log('solve')

    def step(self, tao, x, g, s):
        self._log('step')
        tao.computeGradient(x,g)
        g.copy(s)
        s.scale(-1.0)

    def preStep(self, tao):
        self._log('preStep')

    def postStep(self, tao):
        self._log('postStep')

    def monitor(self, tao):
        self._log('monitor')

class TestTaoPython(unittest.TestCase):

    def setUp(self):
        self.tao = PETSc.TAO()
        self.tao.createPython(MyTao(), comm=PETSc.COMM_SELF)
        ctx = self.tao.getPythonContext()
        self.assertEqual(getrefcount(ctx),  3)
        self.assertEqual(ctx.log['create'], 1)
        self.nsolve = 0

    def tearDown(self):
        ctx = self.tao.getPythonContext()
        self.assertEqual(getrefcount(ctx), 3)
        self.assertTrue('destroy' not in ctx.log)
        self.tao.destroy()
        self.tao = None
        PETSc.garbage_cleanup()
        self.assertEqual(ctx.log['destroy'], 1)
        self.assertEqual(getrefcount(ctx),   2)

    def testGetType(self):
        ctx = self.tao.getPythonContext()
        pytype = "{0}.{1}".format(ctx.__module__, type(ctx).__name__)
        self.assertTrue(self.tao.getPythonType() == pytype)

    def testSolve(self):
        tao = self.tao
        ctx = tao.getPythonContext()
        x = PETSc.Vec().create(tao.getComm())
        x.setType('standard')
        x.setSizes(2)
        y1 = x.duplicate()
        y2 = x.duplicate()
        tao.setObjective(Objective())
        tao.setGradient(Gradient(),None)
        tao.setMonitor(ctx.monitor)
        tao.setFromOptions()
        tao.setMaximumIterations(3)
        tao.setSolution(x)

        # Call the solve method of MyTAO
        x.set(0.5)
        tao.solve()
        n = tao.getIterationNumber()
        self.assertTrue(n == 0)

        # Call the default solve method and use step of MyTAO
        ctx.solve = None
        x.set(0.5)
        tao.solve()
        n = tao.getIterationNumber()
        self.assertTrue(n == 3)
        x.copy(y1)

        # Call the default solve method with the default step method
        ctx.step = None
        x.set(0.5)
        tao.solve()
        n = tao.getIterationNumber()
        self.assertTrue(n == 3)
        x.copy(y2)

        self.assertTrue(y1.equal(y2))
        self.assertTrue(ctx.log['monitor'] == 2*(n+1))
        self.assertTrue(ctx.log['preStep'] == 2*n)
        self.assertTrue(ctx.log['postStep'] == 2*n)
        self.assertTrue(ctx.log['solve'] == 1)
        self.assertTrue(ctx.log['setUp'] == 1)
        self.assertTrue(ctx.log['setFromOptions'] == 1)
        self.assertTrue(ctx.log['step'] == n)
        tao.cancelMonitor()

# --------------------------------------------------------------------

import numpy
if numpy.iscomplexobj(PETSc.ScalarType()):
    del TestTaoPython

if __name__ == '__main__':
    unittest.main()

# --------------------------------------------------------------------
