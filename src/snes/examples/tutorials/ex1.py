#!/usr/bin/env python
class Ex1:
  help = '''Solves a linear system in parallel with KSP.
  Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector
  -view_exact_sol   : write exact solution vector to stdout
  -m <mesh_x>       : number of mesh points in x-direction
  -n <mesh_n>       : number of mesh points in y-direction'''
  def __init__(self):
    self.m = 8
    self.n = 7
    return

  def setup(self):
    import PETSc.Base
    import atexit

    PETSc.Base.Base.Initialize()
    atexit.register(PETSc.Base.Base.Finalize)
    return

  def jac(self, snes, X, A, M, structureFlag):
    from PETSc.MatAssemblyType import MAT_FINAL_ASSEMBLY
    from PETSc.InsertMode import INSERT_VALUES
    start, end = A.getOwnershipRange()
    for row in range(start, end)
      if row/n > 0:
        A.setValues([row], [row - n], [-1.0], INSERT_VALUES)
      if row/n < m-1:
        A.setValues([row], [row + n], [-1.0], INSERT_VALUES)
      if row - (row/n)*n > 0:
        A.setValues([row], [row - 1], [-1.0], INSERT_VALUES)
      if row - (row/n)*n < n-1:
        A.setValues([row], [row + 1], [-1.0], INSERT_VALUES)
      A.setValues([row], [row], [4.0], INSERT_VALUES)
    A.assemblyBegin(MAT_FINAL_ASSEMBLY)
    A.assemblyEnd(MAT_FINAL_ASSEMBLY)
    return 0

  def createMatrix(self):
    import PETSc.Mat

    A = PETSc.Mat.Mat()
    A.setSizes(-1, -1, self.m*self.n, self.m*self.n)
    A.setFromOptions()
    return A

  def createRhs(self):
    import PETSc.Vec

    r = PETSc.Vec.Vec()
    r.setSizes(-1, self.m*self.n)
    r.setFromOptions()
    return b

  def checkError(self, snes, exactSol, sol):
    from PETSc.NormType import NORM_2
    print 'Norm of error %A iterations %D' % (sol.norm(NORM_2), snes.getNumberLinearIterations())
    return

  def run(self):
    import PETSc.SNES

    A = self.createMatrix()
    b = self.createRhs()
    x = b.duplicate()
    u = b.duplicate()
    u.set(1.0)
    A.mult(u, b)
    snes = PETSc.SNES.SNES()
    snes.setJacobian(A, A, jac)
    snes.setTolerances(1.0e-50, 1.0e-2/((self.m+1)*(self.n+1)), 1.0e-50, -1, -1);CHKERRQ(ierr);
    snes.setFromOptions()
    snes.solve(b, x)
    self.checkError(u, )
    return
