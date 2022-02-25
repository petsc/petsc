# --------------------------------------------------------------------

from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------
class Objective:
    def __call__(self, tao, x):
        return (x[0] - 2.0)**2 + (x[1] - 2.0)**2 - 2.0*(x[0] + x[1])

class Gradient:
    def __call__(self, tao, x, g):
        g[0] = 2.0*(x[0] - 2.0) - 2.0
        g[1] = 2.0*(x[1] - 2.0) - 2.0
        g.assemble()

class EqConstraints:
    def __call__(self, tao, x, c):
        c[0] = x[0]**2 + x[1] - 2.0
        c.assemble()

class EqJacobian:
    def __call__(self, tao, x, J, P):
        P[0,0] = 2.0*x[0]
        P[0,1] = 1.0
        P.assemble()
        if J != P: J.assemble()

class BaseTestTAO(object):

    COMM = None

    def setUp(self):
        self.tao = PETSc.TAO().create(comm=self.COMM)

    def tearDown(self):
        self.tao = None

    def testSetRoutinesToNone(self):
        tao = self.tao
        objective, gradient, objgrad = None, None, None
        constraint, varbounds = None, None
        hessian, jacobian = None, None
        tao.setObjective(objective)
        tao.setGradient(gradient,None)
        tao.setVariableBounds(varbounds)
        tao.setObjectiveGradient(objgrad,None)
        tao.setConstraints(constraint)
        tao.setHessian(hessian)
        tao.setJacobian(jacobian)

    def testGetVecsAndMats(self):
        tao = self.tao
        x = tao.getSolution()
        (g, _) = tao.getGradient()
        l, u = tao.getVariableBounds()
        r = None#tao.getConstraintVec()
        H, HP = None,None#tao.getHessianMat()
        J, JP = None,None#tao.getJacobianMat()
        for o in [x, g, r, l, u ,H, HP, J, JP,]:
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
        tao.setGradient(Gradient(),None)
        tao.setEqualityConstraints(EqConstraints(),c)
        tao.setJacobianEquality(EqJacobian(),J,J)
        tao.setSolution(x)
        tao.setType(PETSc.TAO.Type.ALMM)
        tao.setTolerances(gatol=1.e-4)
        tao.setFromOptions()
        tao.solve()
        self.assertAlmostEqual(abs(x[0]**2 + x[1] - 2.0), 0.0, places=4)

# --------------------------------------------------------------------

class TestTAOSelf(BaseTestTAO, unittest.TestCase):
    COMM = PETSc.COMM_SELF

class TestTAOWorld(BaseTestTAO, unittest.TestCase):
    COMM = PETSc.COMM_WORLD

# --------------------------------------------------------------------

import numpy
if numpy.iscomplexobj(PETSc.ScalarType()):
    del BaseTestTAO
    del TestTAOSelf
    del TestTAOWorld

if __name__ == '__main__':
    unittest.main()
