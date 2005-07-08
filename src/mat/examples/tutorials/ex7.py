#!/usr/bin/env python
class Ex1:
  '''Checks that getValues() operates correctly'''
  def __init__(self):
    return

  def setup(self):
    import PETSc.Base
    import atexit

    PETSc.Base.Base.Initialize()
    atexit.register(PETSc.Base.Base.Finalize)
    return

  def createMatrix(self, M, N):
    from PETSc.PetscConstants import PETSC_DECIDE
    from PETSc.InsertMode import INSERT_VALUES
    from PETSc.MatAssemblyType import MAT_FINAL_ASSEMBLY
    import PETSc.Mat

    A = PETSc.Mat.Mat()
    A.setSizes(PETSC_DECIDE, PETSC_DECIDE, M, N)
    A.setFromOptions()
    for row in range(M):
      A.setValues([row], range(N), [(row*N + c)*10.0 for c in range(N)], INSERT_VALUES)
    A.assemblyBegin(MAT_FINAL_ASSEMBLY)
    A.assemblyEnd(MAT_FINAL_ASSEMBLY)
    return A

  def run(self, M = 10, N = 10):
    '''We must be careful here to create a Numeric array to pass into getValues() to avoid a copy'''
    import Numeric

    self.setup()
    A = self.createMatrix(M, N)
    values = Numeric.zeros((1,), 'd')
    for i, j in zip([1, 7, 2], [9, 5, 1]):
      A.getValues([i], [j], values)
      if not values == [(i*N + j)*10.0]:
        raise RuntimeError ('Invalid matrix element(%d, %d) %g should be %g' % (i, j, values[0], (i*N + j)*10.0))
    return

if __name__ == '__main__':
  import os,re,sys
  if '__file__' in globals():
    petscDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
  else:
    petscDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
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
