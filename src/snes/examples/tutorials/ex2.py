#!/usr/bin/env python
class Ex1:
  help = '''Solves a nonlinear system in parallel with SNES.
  Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector
  -view_exact_sol   : write exact solution vector to stdout
  -m <mesh_x>       : number of mesh points in x-direction
  -n <mesh_n>       : number of mesh points in y-direction
  -lambda <lambda>  : nonlinear parameter'''
  m = 4
  n = 4
  mylambda = 6

  def __init__(self):
    return

  def setup(self):
    import PETSc.Base
    import atexit

    PETSc.Base.Base.Initialize()
    atexit.register(PETSc.Base.Base.Finalize)
    return

  def setFromOptions(self):
    import PETSc.PetscOptions
    m, flag = PETSc.PetscOptions.PetscOptions.getInt('', '-m')
    if flag:
      Ex1.m = m
    n, flag = PETSc.PetscOptions.PetscOptions.getInt('', '-n')
    if flag:
      Ex1.n = n
    mylambda, flag = PETSc.PetscOptions.PetscOptions.getReal('', '-lambda')
    if flag:
      Ex1.mylambda = mylambda
    if Ex1.mylambda >= 6.81 or Ex1.mylambda <= 0.0:
      raise RuntimeError('Lambda is out of range: '+str(Ex1.mylambda))
    return

  def createMatrix(self):
    from PETSc.PetscConstants import PETSC_DECIDE
    import PETSc.Mat

    A = PETSc.Mat.Mat()
    A.setSizes(PETSC_DECIDE, PETSC_DECIDE, Ex1.m*Ex1.n, Ex1.m*Ex1.n)
    A.setFromOptions()
    return A

  def createRhs(self):
    from PETSc.PetscConstants import PETSC_DECIDE
    import PETSc.Vec

    b = PETSc.Vec.Vec()
    b.setSizes(PETSC_DECIDE, Ex1.m*Ex1.n)
    b.setFromOptions()
    return b

  def formInitialGuess(self, X):
    from math import sqrt
    start, end = X.getPetscMap().getLocalRange()
    m = Ex1.m
    n = Ex1.n
    hx = 1.0/(m-1)
    hy = 1.0/(n-1)
    temp1 = Ex1.mylambda/(Ex1.mylambda + 1.0)
    x = X.getArray()
    for row in range(start, end):
      # Bottom boundary
      if row < n:
        x[row] = 0.0
      # Top boundary
      elif row >= m*(n-1):
        x[row] = 0.0
      # Left boundary
      elif row - (row/n)*n == 0:
        x[row] = 0.0
      # Right boundary
      elif row - (row/n)*n == n-1:
        x[row] = 0.0
      # Interior
      else:
        i = row/n
        j = row - (row/n)*n
        x[row] = temp1*sqrt(min(min(i, m-i-1)*hx, min(j, n-j-1)*hy))
    X.restoreArray(x)
    X.assemblyBegin()
    X.assemblyEnd()
    return 0

  @staticmethod
  def func(snes, X, F):
    from math import exp
    m = Ex1.m
    n = Ex1.n
    mylambda = Ex1.mylambda
    hx = 1.0/(m-1)
    hy = 1.0/(n-1)
    hxdhy = hx/hy 
    hydhx = hy/hx
    sc = hx*hy*mylambda
    x = X.getArray()
    f = F.getArray()
    start, end = F.getPetscMap().getLocalRange()
    for row in range(start, end):
      # Bottom boundary
      if row < n:
        f[row] = x[row]
      # Top boundary
      elif row >= m*(n-1):
        f[row] = x[row]
      # Left boundary
      elif row - (row/n)*n == 0:
        f[row] = x[row]
      # Right boundary
      elif row - (row/n)*n == n-1:
        f[row] = x[row]
      # Interior
      else:
        u      = x[row]
        uxx    = (2.0*u - x[row-1] - x[row+1])*hydhx
        uyy    = (2.0*u - x[row-n] - x[row+n])*hxdhy
        f[row] = uxx + uyy - sc*exp(u)
    X.restoreArray(x)
    F.restoreArray(f)
    F.assemblyBegin()
    F.assemblyEnd()
    return 0

  @staticmethod
  def jac(snes, X, A, M, structureFlag):
    from math import exp
    from PETSc.MatAssemblyType import MAT_FINAL_ASSEMBLY
    from PETSc.InsertMode import INSERT_VALUES
    m = Ex1.m
    n = Ex1.n
    mylambda = Ex1.mylambda
    hx = 1.0/(m-1)
    hy = 1.0/(n-1)
    hxdhy = hx/hy 
    hydhx = hy/hx
    sc = hx*hy*mylambda
    start, end = A.getOwnershipRange()
    x = X.getArray()
    for row in range(start, end):
      # Bottom boundary
      if row < n:
        A.setValues([row], [row], [1.0], INSERT_VALUES)
      # Top boundary
      elif row >= m*(n-1):
        A.setValues([row], [row], [1.0], INSERT_VALUES)
      # Left boundary
      elif row - (row/n)*n == 0:
        A.setValues([row], [row], [1.0], INSERT_VALUES)
      # Right boundary
      elif row - (row/n)*n == n-1:
        A.setValues([row], [row], [1.0], INSERT_VALUES)
      # Interior
      else:
        A.setValues([row], [row - n], [-hxdhy], INSERT_VALUES)
        A.setValues([row], [row + n], [-hxdhy], INSERT_VALUES)
        A.setValues([row], [row - 1], [-hydhx], INSERT_VALUES)
        A.setValues([row], [row + 1], [-hydhx], INSERT_VALUES)
        A.setValues([row], [row], [2.0*(hydhx + hxdhy) - sc*exp(x[row])], INSERT_VALUES)
    X.restoreArray(x)
    A.assemblyBegin(MAT_FINAL_ASSEMBLY)
    A.assemblyEnd(MAT_FINAL_ASSEMBLY)
    return 0

  def solveNonlinear(self, snes, A, r):
    snes.setFunction(r, self.func)
    snes.setJacobian(A, A, self.jac)
    x = r.duplicate()
    self.formInitialGuess(x)
    snes.solve(None, x)
    return x

  def run(self):
    from PETSc.PetscConstants import PETSC_DEFAULT
    import PETSc.SNES

    self.setup()
    self.setFromOptions()
    snes = PETSc.SNES.SNES()
    snes.setFromOptions()
    A = self.createMatrix()
    r = self.createRhs()
    self.solveNonlinear(snes, A, r)
    print 'Number of Newton iterations =', snes.getIterationNumber()
    return

if __name__ == '__main__':
  import os,re,sys
  #petscDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
  petscDir = '/PETSc3/petsc/petsc-dev'
  petscConf = os.path.join(petscDir, 'bmake', 'petscconf')
  if 'PETSC_ARCH' in os.environ:
    petscArch = os.environ['PETSC_ARCH']
  elif os.path.isfile(petscConf):
    archRE = re.compile(r'^PETSC_ARCH=(?P<arch>[\w.\d-]+)[\s]*$')
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
