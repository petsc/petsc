import unittest
from petsc4py import PETSc
from sys import getrefcount
import numpy


# --------------------------------------------------------------------
class Objective:
    def __call__(self, tao, x):
        return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2


class Gradient:
    def __call__(self, tao, x, g):
        g[0] = 2.0 * (x[0] - 1.0)
        g[1] = 2.0 * (x[1] - 2.0)
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
        tao.computeGradient(x, g)
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
        self.assertEqual(getrefcount(self._getCtx()), 2)
        self.assertEqual(self._getCtx().log['create'], 1)
        self.nsolve = 0

    def tearDown(self):
        self.assertEqual(getrefcount(self._getCtx()), 2)
        self.assertTrue('destroy' not in self._getCtx().log)
        ctx = self._getCtx()
        self.tao.destroy()
        self.tao = None
        PETSc.garbage_cleanup()
        self.assertEqual(ctx.log['destroy'], 1)

    def testGetType(self):
        ctx = self.tao.getPythonContext()
        pytype = f'{ctx.__module__}.{type(ctx).__name__}'
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
        tao.setGradient(Gradient(), None)
        tao.setMonitor(ctx.monitor)
        tao.setFromOptions()
        tao.setMaximumIterations(3)

        def _update(tao, it, cnt):
            cnt += 1

        cnt_up = numpy.array(0)
        tao.setUpdate(_update, (cnt_up,))
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
        self.assertGreater(tao.getConvergedReason(), 0)
        self.assertTrue(n in [2, 3])
        self.assertAlmostEqual(x[0], 1.0)
        self.assertAlmostEqual(x[1], 2.0)

        # Call the default solve method with the default step method
        ctx.step = None
        x.set(0.5)
        tao.solve()
        n = tao.getIterationNumber()
        self.assertGreater(tao.getConvergedReason(), 0)
        self.assertTrue(n in [2, 3])
        self.assertAlmostEqual(x[0], 1.0)
        self.assertAlmostEqual(x[1], 2.0)

        self.assertTrue(y1.equal(y2))
        self.assertTrue(ctx.log['monitor'] == 2 * (n + 1))
        self.assertTrue(ctx.log['preStep'] == 2 * n)
        self.assertTrue(ctx.log['postStep'] == 2 * n)
        self.assertTrue(ctx.log['solve'] == 1)
        self.assertTrue(ctx.log['setUp'] == 1)
        self.assertTrue(ctx.log['setFromOptions'] == 1)
        self.assertTrue(ctx.log['step'] == n)
        self.assertEqual(cnt_up, 2 * n)
        tao.cancelMonitor()

    def _getCtx(self):
        return self.tao.getPythonContext()


class MyGradientDescent:
    def __init__(self):
        self._ls = None

    def create(self, tao):
        self._ls = PETSc.TAOLineSearch().create(comm=PETSc.COMM_SELF)
        self._ls.useTAORoutine(tao)
        self._ls.setType(PETSc.TAOLineSearch.Type.UNIT)
        self._ls.setInitialStepLength(0.2)

    def destroy(self, tao):
        self._ls.destroy()

    def setUp(self, tao):
        pass

    def solve(self, tao):
        x = tao.getSolution()
        gradient = tao.getGradient()[0]
        search_direction = gradient.copy()
        for it in range(tao.getMaximumIterations()):
            tao.setIterationNumber(it)

            # search_direction = -gradient
            tao.computeGradient(x, gradient)
            gradient.copy(search_direction)
            search_direction.scale(-1)

            # x = x + .2 search_direction
            f, s, reason = self._ls.apply(x, gradient, search_direction)

            tao.monitor(f=f, res=gradient.norm())

            if reason < 0:
                raise RuntimeError('LS failed.')

            if tao.checkConverged() > 0:
                break

    def step(self, tao, x, g, s):
        raise RuntimeError('Should only be called by builtin solve.')

    def preStep(self, tao):
        raise RuntimeError('Should only be called by builtin solve.')

    def postStep(self, tao):
        raise RuntimeError('Should only be called by builtin solve.')


class TestTaoPythonOptimiser(unittest.TestCase):
    def setUp(self):
        self.tao = PETSc.TAO()
        self.tao.createPython(MyGradientDescent(), comm=PETSc.COMM_SELF)

    def tearDown(self):
        self.tao.destroy()
        self.tao = None

    def testSolve(self):
        tao = self.tao

        opts = PETSc.Options('test_tao_python_optimiser_')
        opts['tao_max_it'] = 100
        opts['tao_gatol'] = 1e-6

        tao.setOptionsPrefix('test_tao_python_optimiser_')
        tao.setFromOptions()

        x = PETSc.Vec().createSeq(2, comm=tao.getComm())
        x.set(0.5)

        tao.setSolution(x)
        tao.setObjective(Objective())
        tao.setGradient(Gradient(), x.copy())

        tao.solve()

        self.assertEqual(tao.getMaximumIterations(), 100)
        self.assertAlmostEqual(tao.getTolerances()[0], 1e-6)
        self.assertGreater(tao.getIterationNumber(), 0)
        self.assertGreater(tao.getConvergedReason(), 0)
        self.assertAlmostEqual(x[0], 1.0, places=5)
        self.assertAlmostEqual(x[1], 2.0, places=5)
        self.assertGreater(tao.getObjectiveValue(), 0)
        self.assertAlmostEqual(tao.getObjectiveValue(), 0, places=5)


# --------------------------------------------------------------------

if numpy.iscomplexobj(PETSc.ScalarType()):
    del TestTaoPython
    del TestTaoPythonOptimiser

if __name__ == '__main__':
    unittest.main()

# --------------------------------------------------------------------
