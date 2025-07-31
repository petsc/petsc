# --------------------------------------------------------------------

from math import sqrt
from petsc4py import PETSc
import unittest
import numpy


# --------------------------------------------------------------------
class Objective:
    def __call__(self, tao, x):
        return (x[0] - 2.0) ** 2 + (x[1] - 2.0) ** 2 - 2.0 * (x[0] + x[1])


class Gradient:
    def __call__(self, tao, x, g):
        g[0] = 2.0 * (x[0] - 2.0) - 2.0
        g[1] = 2.0 * (x[1] - 2.0) - 2.0
        g.assemble()


class EqConstraints:
    def __call__(self, tao, x, c):
        c[0] = x[0] ** 2 + x[1] - 2.0
        c.assemble()


class EqJacobian:
    def __call__(self, tao, x, J, P):
        P[0, 0] = 2.0 * x[0]
        P[0, 1] = 1.0
        P.assemble()
        if J != P:
            J.assemble()


class InEqConstraints:
    def __call__(self, tao, x, c):
        c[0] = x[1] - x[0] ** 2
        c.assemble()


class InEqJacobian:
    def __call__(self, tao, x, J, P):
        P[0, 0] = -2.0 * x[0]
        P[0, 1] = 1.0
        P.assemble()
        if J != P:
            J.assemble()


class BaseTestTAO:
    COMM = None

    def setUp(self):
        self.tao = PETSc.TAO().create(comm=self.COMM)

    def tearDown(self):
        self.tao = None
        PETSc.garbage_cleanup()

    def testSetRoutinesToNone(self):
        tao = self.tao
        objective, gradient, objgrad = None, None, None
        constraint, varbounds = None, None
        hessian, jacobian = None, None
        tao.setObjective(objective)
        tao.setGradient(gradient, None)
        tao.setVariableBounds(varbounds)
        tao.setObjectiveGradient(objgrad, None)
        tao.setConstraints(constraint)
        tao.setHessian(hessian)
        tao.setJacobian(jacobian)

    def testGetVecsAndMats(self):
        tao = self.tao
        x = tao.getSolution()
        (g, _) = tao.getGradient()
        low, up = tao.getVariableBounds()
        r = None  # tao.getConstraintVec()
        H, HP = None, None  # tao.getHessianMat()
        J, JP = None, None  # tao.getJacobianMat()
        for o in [
            x,
            g,
            r,
            low,
            up,
            H,
            HP,
            J,
            JP,
        ]:
            self.assertFalse(o)

    def testGetKSP(self):
        ksp = self.tao.getKSP()
        self.assertFalse(ksp)

    def testEqualityConstraints(self):
        if self.tao.getComm().Get_size() > 1:
            return
        tao = self.tao

        x = PETSc.Vec().create(tao.getComm())
        x.setType('standard')
        x.setSizes(2)
        c = PETSc.Vec().create(tao.getComm())
        c.setSizes(1)
        c.setType(x.getType())
        J = PETSc.Mat().create(tao.getComm())
        J.setSizes([1, 2])
        J.setType(PETSc.Mat.Type.DENSE)
        J.setUp()

        tao.setObjective(Objective())
        tao.setGradient(Gradient(), None)
        tao.setEqualityConstraints(EqConstraints(), c)
        tao.setJacobianEquality(EqJacobian(), J, J)
        tao.setSolution(x)
        tao.setType(PETSc.TAO.Type.ALMM)
        tao.setALMMType(PETSc.TAO.ALMMType.PHR)
        tao.setTolerances(gatol=1.0e-4)
        tao.setFromOptions()
        tao.solve()
        self.assertTrue(tao.getALMMType() == PETSc.TAO.ALMMType.PHR)
        self.assertAlmostEqual(abs(x[0] ** 2 + x[1] - 2.0), 0.0, places=4)
        self.assertAlmostEqual(x[0], 0.7351392590499015014254200465, places=4)
        self.assertAlmostEqual(x[1], 1.4595702698035618134357683666, places=4)
        self.assertTrue(tao.getObjective() is not None)

        c, g = tao.getEqualityConstraints()
        c_eval = c.copy()
        g[0](tao, x, c_eval, *g[1], **g[2])
        self.assertTrue(c.equal(c_eval))

        J, Jpre, Jg = tao.getJacobianEquality()
        Jg[0](tao, x, J, Jpre, *Jg[1], **Jg[2])
        self.assertTrue(J.equal(Jpre))

    def testInequlityConstraints(self):
        if self.tao.getComm().Get_size() > 1:
            return
        tao = self.tao

        x = PETSc.Vec().create(tao.getComm())
        x.setType('standard')
        x.setSizes(2)
        c = PETSc.Vec().create(tao.getComm())
        c.setSizes(1)
        c.setType(x.getType())
        J = PETSc.Mat().create(tao.getComm())
        J.setSizes([1, 2])
        J.setType(PETSc.Mat.Type.DENSE)
        J.setUp()

        tao.setObjective(Objective())
        tao.setGradient(Gradient(), None)
        tao.setInequalityConstraints(InEqConstraints(), c)
        tao.setJacobianInequality(InEqJacobian(), J, J)
        tao.setSolution(x)
        tao.setType(PETSc.TAO.Type.ALMM)
        tao.setALMMType(PETSc.TAO.ALMMType.CLASSIC)
        tao.setTolerances(gatol=1.0e-4)
        tao.setFromOptions()
        tao.solve()

        self.assertTrue(tao.getALMMType() == PETSc.TAO.ALMMType.CLASSIC)
        self.assertTrue(x[1] - x[0] ** 2 >= -1.0e-4)
        self.assertAlmostEqual(x[0], 0.5 + sqrt(7) / 2, places=4)
        self.assertAlmostEqual(x[1], 2 + sqrt(7) / 2, places=4)

        c, h = tao.getInequalityConstraints()
        c_eval = c.copy()
        h[0](tao, x, c_eval, *h[1], **h[2])
        self.assertTrue(c.equal(c_eval))

        J, Jpre, Jh = tao.getJacobianInequality()
        Jh[0](tao, x, J, Jpre, *Jh[1], **Jh[2])
        self.assertTrue(J.equal(Jpre))

    def testBNCG(self):
        if self.tao.getComm().Get_size() > 1:
            return
        tao = self.tao

        x = PETSc.Vec().create(tao.getComm())
        x.setType('standard')
        x.setSizes(2)
        xl = PETSc.Vec().create(tao.getComm())
        xl.setType('standard')
        xl.setSizes(2)
        xl.set(0.0)
        xu = PETSc.Vec().create(tao.getComm())
        xu.setType('standard')
        xu.setSizes(2)
        xu.set(2.0)
        tao.setVariableBounds((xl, xu))
        tao.setObjective(Objective())
        tao.setGradient(Gradient(), None)
        tao.setSolution(x)
        tao.setType(PETSc.TAO.Type.BNCG)
        tao.setTolerances(gatol=1.0e-4)
        ls = tao.getLineSearch()
        ls.setType(PETSc.TAOLineSearch.Type.UNIT)
        tao.setFromOptions()
        tao.solve()
        self.assertAlmostEqual(x[0], 2.0, places=4)
        self.assertAlmostEqual(x[1], 2.0, places=4)

    def templateBQNLS(self, lmvm_setup):
        if self.tao.getComm().Get_size() > 1:
            return
        tao = self.tao

        x = PETSc.Vec().create(tao.getComm())
        x.setType('standard')
        x.setSizes(2)
        xl = PETSc.Vec().create(tao.getComm())
        xl.setType('standard')
        xl.setSizes(2)
        xl.set(0.0)
        xu = PETSc.Vec().create(tao.getComm())
        xu.setType('standard')
        xu.setSizes(2)
        xu.set(2.0)
        tao.setVariableBounds((xl, xu))
        tao.setObjective(Objective())
        tao.setGradient(Gradient(), None)
        tao.setSolution(x)
        tao.setType(PETSc.TAO.Type.BQNLS)
        tao.setTolerances(gatol=1.0e-4)

        H = PETSc.Mat()
        if lmvm_setup == 'dense' or lmvm_setup == 'ksp':
            H.createDense((2, 2), comm=tao.getComm())
            H[0, 0] = 2
            H[0, 1] = 0
            H[1, 0] = 0
            H[1, 1] = 2
            H.assemble()
        elif lmvm_setup == 'diagonal':
            H_vec = PETSc.Vec().createSeq(2)
            H_vec[0] = 2
            H_vec[1] = 2
            H_vec.assemble()
            H.createDiagonal(H_vec)
            H.assemble()

        if lmvm_setup == 'dense' or lmvm_setup == 'diagonal':
            tao.getLMVMMat().setLMVMJ0(H)
        elif lmvm_setup == 'ksp':
            lmvm_ksp = PETSc.KSP().create(tao.getComm())
            lmvm_ksp.setType(PETSc.KSP.Type.CG)
            lmvm_ksp.setOperators(H)
            tao.getLMVMMat().setLMVMJ0KSP(lmvm_ksp)

        tao.setFromOptions()
        tao.solve()
        if lmvm_setup == 'dense':
            self.assertEqual(tao.getIterationNumber(), 1)
        self.assertAlmostEqual(x[0], 2.0, places=4)
        self.assertAlmostEqual(x[1], 2.0, places=4)

        if lmvm_setup == 'dense' or lmvm_setup == 'diagonal':
            self.assertTrue(tao.getLMVMMat().getLMVMJ0().equal(H))
        elif lmvm_setup == 'ksp':
            self.assertTrue(
                tao.getLMVMMat().getLMVMJ0KSP().getType() == PETSc.KSP.Type.CG
            )

    def testBQNLS_dense(self):
        self.templateBQNLS('dense')

    def testBQNLS_ksp(self):
        self.templateBQNLS('ksp')

    def testBQNLS_diagonal(self):
        self.templateBQNLS('diagonal')


# --------------------------------------------------------------------


class TestTAOSelf(BaseTestTAO, unittest.TestCase):
    COMM = PETSc.COMM_SELF


class TestTAOWorld(BaseTestTAO, unittest.TestCase):
    COMM = PETSc.COMM_WORLD


# --------------------------------------------------------------------


if numpy.iscomplexobj(PETSc.ScalarType()):
    del BaseTestTAO
    del TestTAOSelf
    del TestTAOWorld

if __name__ == '__main__':
    unittest.main()
