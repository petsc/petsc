# --------------------------------------------------------------------

from petsc4py import PETSc
import unittest
import numpy


# --------------------------------------------------------------------

class HalfL2SquaredObjective:
    """f(x) = 0.5 * ||x - p||^2"""
    def __init__(self, p):
        self.p = p

    def __call__(self, tao, x):
        diff = x - self.p
        val = 0.5 * diff.dot(diff)
        diff.destroy()
        return val


class HalfL2SquaredGradient:
    """g(x) = x - p"""
    def __init__(self, p):
        self.p = p

    def __call__(self, tao, x, g):
        x.copy(g)
        g.axpy(-1.0, self.p)


# --------------------------------------------------------------------

class BaseTestTAOWithTAOTerm:
    COMM = None

    def setUp(self):
        self.tao = PETSc.TAO().create(comm=self.COMM)

    def tearDown(self):
        self.tao = None
        PETSc.garbage_cleanup()

    def testTwoHalfL2SquaredTerms(self):
        if self.tao.getComm().Get_size() > 1:
            return
        tao = self.tao
        n = 2
        target = [3.0, 3.0]
        scale1, scale2 = 0.3, 0.7

        # Parameters vector (shift target)
        p = PETSc.Vec().create(tao.getComm())
        p.setType('standard')
        p.setSizes(n)
        p.setArray(target)
        p.assemble()

        # Solution vector
        x = PETSc.Vec().create(tao.getComm())
        x.setType('standard')
        x.setSizes(n)
        x.set(0.0)

        # Two HALFL2SQUARED terms with different scales but same params
        # Total objective: 0.3 * 0.5*||x - p||^2 + 0.7 * 0.5*||x - p||^2
        #                = 0.5*||x - p||^2, minimum at x = p
        term1 = PETSc.TAOTerm().create(comm=self.COMM)
        term1.setType(PETSc.TAOTerm.Type.HALFL2SQUARED)
        term1.setSolutionTemplate(x)
        tao.addTerm("term1_", scale1, term1, p)

        term2 = PETSc.TAOTerm().create(comm=self.COMM)
        term2.setType(PETSc.TAOTerm.Type.HALFL2SQUARED)
        term2.setSolutionTemplate(x)
        tao.addTerm("term2_", scale2, term2, p)

        tao.setSolution(x)
        tao.setType(PETSc.TAO.Type.LMVM)
        tao.setTolerances(gatol=1.0e-8)
        tao.setFromOptions()
        tao.solve()

        self.assertAlmostEqual(x[0], target[0], places=6)
        self.assertAlmostEqual(x[1], target[1], places=6)

        term1.destroy()
        term2.destroy()
        p.destroy()
        x.destroy()

    def testCallbackVsSplitTerms(self):
        """Verify that callback-based TAO and TaoTerm-based TAO give the same solution."""
        if self.tao.getComm().Get_size() > 1:
            return
        tao = self.tao
        n = 4
        target = [1.0, -2.0, 3.0, -4.0]
        scale1, scale2 = 0.25, 0.75

        p = PETSc.Vec().create(tao.getComm())
        p.setType('standard')
        p.setSizes(n)
        p.setArray(target)
        p.assemble()

        # Solve with callbacks (manual objective and gradient)
        x_cb = PETSc.Vec().create(tao.getComm())
        x_cb.setType('standard')
        x_cb.setSizes(n)
        x_cb.set(0.0)

        tao.setObjective(HalfL2SquaredObjective(p))
        tao.setGradient(HalfL2SquaredGradient(p), None)
        tao.setSolution(x_cb)
        tao.setType(PETSc.TAO.Type.LMVM)
        tao.setTolerances(gatol=1.0e-8)
        tao.setFromOptions()
        tao.solve()

        # Solve with two split TaoTerms
        x_split = PETSc.Vec().create(tao.getComm())
        x_split.setType('standard')
        x_split.setSizes(n)
        x_split.set(0.0)

        tao_split = PETSc.TAO().create(comm=self.COMM)
        term1 = PETSc.TAOTerm().create(comm=self.COMM)
        term1.setType(PETSc.TAOTerm.Type.HALFL2SQUARED)
        term1.setSolutionTemplate(x_split)
        tao_split.addTerm("split1_", scale1, term1, p)

        term2 = PETSc.TAOTerm().create(comm=self.COMM)
        term2.setType(PETSc.TAOTerm.Type.HALFL2SQUARED)
        term2.setSolutionTemplate(x_split)
        tao_split.addTerm("split2_", scale2, term2, p)
        tao_split.setSolution(x_split)
        tao_split.setType(PETSc.TAO.Type.LMVM)
        tao_split.setTolerances(gatol=1.0e-8)
        tao_split.setFromOptions()
        tao_split.solve()

        # Solutions should match
        for i in range(n):
            self.assertAlmostEqual(x_cb[i], x_split[i], places=6)
            self.assertAlmostEqual(x_split[i], target[i], places=6)

        term1.destroy()
        term2.destroy()
        tao_split.destroy()
        p.destroy()
        x_cb.destroy()
        x_split.destroy()


# --------------------------------------------------------------------


class TestTAOWithTAOTermSelf(BaseTestTAOWithTAOTerm, unittest.TestCase):
    COMM = PETSc.COMM_SELF


class TestTAOWithTAOTermWorld(BaseTestTAOWithTAOTerm, unittest.TestCase):
    COMM = PETSc.COMM_WORLD


# --------------------------------------------------------------------


if numpy.iscomplexobj(PETSc.ScalarType()):
    del BaseTestTAOWithTAOTerm
    del TestTAOWithTAOTermSelf
    del TestTAOWithTAOTermWorld

if __name__ == '__main__':
    unittest.main()
