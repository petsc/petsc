# --------------------------------------------------------------------

from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------

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
        tao.setGradient(gradient)
        tao.setVariableBounds(varbounds)
        tao.setObjectiveGradient(objgrad)
        tao.setConstraints(constraint)
        tao.setHessian(hessian)
        tao.setJacobian(jacobian)

    def testGetVecsAndMats(self):
        tao = self.tao
        x = tao.getSolution()
        g = tao.getGradient()
        l, u = tao.getVariableBounds()
        r = None#tao.getConstraintVec()
        H, HP = None,None#tao.getHessianMat()
        J, JP = None,None#tao.getJacobianMat()
        for o in [x, g, r, l, u ,H, HP, J, JP,]:
            self.assertFalse(o)

    def testGetKSP(self):
        ksp = self.tao.getKSP()
        self.assertFalse(ksp)

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
