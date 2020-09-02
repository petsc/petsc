from petsc4py import PETSc
import unittest

import numpy as N

def mkmat(n, mtype, opts):
    A = PETSc.Mat().create(PETSc.COMM_SELF)
    A.setSizes([n,n])
    A.setType(mtype)
    A.setUp()
    for o in opts:
        A.setOption(o, True)
    return A

def mksys_diag(n, mtype, opts):
    A = mkmat(n, mtype, opts)
    x, b = A.createVecs()
    for i in range(n):
        A[i,i] = i+1
        x[i]   = 1.0/(i+1)
        b[i]   = 1
    A.assemble()
    x.assemble()
    b.assemble()
    return A, x, b

def mksys_poi2(n, mtype, opts):
    A = mkmat(n, mtype, opts)
    x, b = A.createVecs()
    for i in range(n):
        if i == 0:
            cols = [i, i+1]
            vals = [2, -1]
        elif i == n-1:
            cols = [i-1, i]
            vals = [-1,  2]
        else:
            cols = [i-1, i, i+1]
            vals = [-1,  2, -1]
        A[i,cols] = vals
        x[i]   = i+1
        b[i]   = 0
    A.assemble()
    x.assemble()
    b.assemble()
    A.mult(x,b)
    return A, x, b

class BaseTestMatFactor(object):

    MKSYS = None
    MTYPE = None
    MOPTS = ()

    def setUp(self):
        A, x, b = self.MKSYS(10, self.MTYPE, self.MOPTS)
        self.A = A
        self.x = x
        self.b = b

    def tearDown(self):
        self.A.setUnfactored()
        self.A.destroy(); self.A = None
        self.x.destroy(); self.x = None
        self.b.destroy(); self.b = None

class BaseTestMatFactorLU(BaseTestMatFactor):

    def testFactorLU(self):
        r, c = self.A.getOrdering("nd")
        self.A.reorderForNonzeroDiagonal(r, c)
        self.A.factorLU(r,c,{'zeropivot':1e-5})
        x = self.x.duplicate()
        self.A.solve(self.b, x)
        x.axpy(-1, self.x)
        self.assertTrue(x.norm() < 1e-3)

class BaseTestMatFactorILU(BaseTestMatFactor):

    def testFactorILU(self):
        r, c = self.A.getOrdering("natural")
        self.A.factorILU(r,c,{'levels':0})
        x = self.x.duplicate()
        self.A.solve(self.b, x)
        x.axpy(-1, self.x)
        self.assertTrue(x.norm() < 1e-3)

## class BaseTestMatFactorILUDT(BaseTestMatFactor):
## 
##     def testFactorILUDT(self):
##         r, c = self.A.getOrdering("natural")
##         self.A = self.A.factorILUDT(r,c)
##         x = self.x.duplicate()
##         self.A.solve(self.b, x)
##         x.axpy(-1, self.x)
##         self.assertTrue(x.norm() < 1e-3)
## 
class BaseTestMatFactorChol(BaseTestMatFactor):

    def testFactorChol(self):
        r, c = self.A.getOrdering("natural")
        self.A.factorCholesky(r)
        x = self.x.duplicate()
        self.A.solve(self.b, x)
        x.axpy(-1, self.x)
        self.assertTrue(x.norm() < 1e-3)

class BaseTestMatFactorICC(BaseTestMatFactor):

    def testFactorICC(self):
        r, c = self.A.getOrdering("natural")
        self.A.factorICC(r)
        x = self.x.duplicate()
        self.A.solve(self.b, x)
        x.axpy(-1, self.x)
        self.assertTrue(x.norm() < 1e-3)


# --------------------------------------------------------------------

class TestMatFactorA1(BaseTestMatFactorLU,
                      BaseTestMatFactorChol,
                      unittest.TestCase):
    MKSYS = staticmethod(mksys_diag)
    MTYPE = PETSc.Mat.Type.SEQDENSE

class TestMatFactorA2(BaseTestMatFactorLU,
                      BaseTestMatFactorChol,
                      unittest.TestCase):
    MKSYS = staticmethod(mksys_poi2)
    MTYPE = PETSc.Mat.Type.SEQDENSE

# ---

class TestMatFactorB1(BaseTestMatFactorLU,
                      BaseTestMatFactorILU,
                      ## BaseTestMatFactorILUDT,
                      unittest.TestCase):
    MKSYS = staticmethod(mksys_diag)
    MTYPE = PETSc.Mat.Type.SEQAIJ

class TestMatFactorB2(BaseTestMatFactorLU,
                      BaseTestMatFactorILU,
                      ## BaseTestMatFactorILUDT,
                      unittest.TestCase):
    MKSYS = staticmethod(mksys_poi2)
    MTYPE = PETSc.Mat.Type.SEQAIJ

# ---

class TestMatFactorC1(BaseTestMatFactorLU,
                      BaseTestMatFactorILU,
                      unittest.TestCase):
    MKSYS = staticmethod(mksys_diag)
    MTYPE = PETSc.Mat.Type.SEQBAIJ

class TestMatFactorC2(BaseTestMatFactorLU,
                      BaseTestMatFactorILU,
                      unittest.TestCase):
    MKSYS = staticmethod(mksys_poi2)
    MTYPE = PETSc.Mat.Type.SEQBAIJ

# ---

class TestMatFactorD1(BaseTestMatFactorChol,
                      BaseTestMatFactorICC,
                      unittest.TestCase):
    MKSYS = staticmethod(mksys_diag)
    MTYPE = PETSc.Mat.Type.SEQSBAIJ
    MOPTS = [PETSc.Mat.Option.IGNORE_LOWER_TRIANGULAR]

class TestMatFactorD2(BaseTestMatFactorChol,
                      BaseTestMatFactorICC,
                      unittest.TestCase):
    MKSYS = staticmethod(mksys_poi2)
    MTYPE = PETSc.Mat.Type.SEQSBAIJ
    MOPTS = [PETSc.Mat.Option.IGNORE_LOWER_TRIANGULAR]

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
