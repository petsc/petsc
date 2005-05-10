#!/usr/bin/env python
def func(snes, X, F):
  global A, b
  A.mult(X, F)
  F.AXPY(-1.0, b)
  return 0

def jac(snes, X, A, M, structureFlag):
  from PETSc.MatAssemblyType import MAT_FINAL_ASSEMBLY
  from PETSc.InsertMode import INSERT_VALUES
  start, end = A.getOwnershipRange()
  for row in range(start, end):
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

class Ex1:
  help = '''Solves a linear system in parallel with KSP.
  Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector
  -view_exact_sol   : write exact solution vector to stdout
  -m <mesh_x>       : number of mesh points in x-direction
  -n <mesh_n>       : number of mesh points in y-direction'''
  def __init__(self):
    self.m = 3
    self.n = 3
    return

  def setup(self):
    import PETSc.Base
    import atexit

    PETSc.Base.Base.Initialize()
    atexit.register(PETSc.Base.Base.Finalize)
    return

  def createMatrix(self):
    import PETSc.Mat

    A = PETSc.Mat.Mat()
    A.setSizes(-1, -1, self.m*self.n, self.m*self.n)
    A.setFromOptions()
    return A

  def createRhs(self):
    import PETSc.Vec

    b = PETSc.Vec.Vec()
    b.setSizes(-1, self.m*self.n)
    b.setFromOptions()
    return b

  def checkError(self, snes, u, x):
    from PETSc.NormType import NORM_2
    x.AXPY(-1.0, u)
    print 'Norm of error %g iterations %i' % (x.norm(NORM_2), snes.getNumberLinearIterations())
    return

  def solveLinear(self, snes, A, b):
    jac(snes, None, A, A, None)
    snes.setJacobian(A, A, None)
    x = b.duplicate()
    snes.solve(b, x)
    snes.setRhs(None)
    return x

  def solveNonlinear(self, snes, myA, myb):
    global A, b
    A = myA
    b = myb
    r = b.duplicate()
    snes.setFunction(r, func)
    snes.setJacobian(A, A, jac)
    x = b.duplicate()
    snes.solve(None, x)
    del A
    del b
    return x

  def run(self):
    from PETSc.PetscConstants import PETSC_DEFAULT
    import PETSc.SNES
    global m,n

    self.setup()
    m = self.m
    n = self.n
    snes = PETSc.SNES.SNES()
    snes.setTolerances(1.0e-50, 1.0e-2/((self.m+1)*(self.n+1)), 1.0e-50, PETSC_DEFAULT, PETSC_DEFAULT)
    snes.setFromOptions()
    A = self.createMatrix()
    b = self.createRhs()
    u = b.duplicate()
    u.set(1.0)
    jac(snes, None, A, A, None)
    A.mult(u, b)
    A.zeroEntries()
    self.checkError(snes, u, self.solveLinear(snes, A, b))
    self.checkError(snes, u, self.solveNonlinear(snes, A, b))
    return

if __name__ == '__main__':
  import os,re,sys
  petscDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
  petscConf = os.path.join(petscDir, 'bmake', 'petscconf')
  if 'PETSC_ARCH' in os.environ:
    petscArch = os.environ['PETSC_ARCH']
  elif os.path.isfile(petscConf):
    archRE = re.compile(r'^PETSC_ARCH=(?P<arch>[\w.\d-]+)[\s]*$');
    confFile = file(petscConf)
    for line in confFile.readlines():
      m = archRE.match(line)
      if m:
        petscArch = m.group('arch')
    confFile.close()
  else:
    raise RuntimeError('Could not determine PETSC_ARCH')
  sys.path.append(os.path.join(petscDir, 'lib', petscArch))
  Ex1().run()
