#!/usr/bin/env python
class Ex1:
  '''Resizes dense matrices and uses MatGetSubMatrix() and MatGetArray()'''
  def __init__(self):
    return

  def setup(self):
    import PETSc.Base
    import atexit

    PETSc.Base.Base.Initialize()
    atexit.register(PETSc.Base.Base.Finalize)
    return

  def createMatrices(self, m,n,M, N):
    from PETSc.PetscConstants import PETSC_DECIDE
    from PETSc.InsertMode import INSERT_VALUES
    from PETSc.MatAssemblyType import MAT_FINAL_ASSEMBLY
    import PETSc.Mat

    # allocate a big matrix A, we will only use the first
    # certain number of rows and columns, which will increase over time
    A = PETSc.Mat.Mat()
    A.setSizes(M,N, PETSC_DECIDE, PETSC_DECIDE)
    A.setType('seqdense')
    A.assemblyBegin(MAT_FINAL_ASSEMBLY)
    A.assemblyEnd(MAT_FINAL_ASSEMBLY)

    B = PETSc.Mat.Mat()
    B.setSizes(m,n,PETSC_DECIDE,PETSC_DECIDE)
    B.setType('seqdense')
  
    return A,B

  def run(self, m = 5, n = 5,M = 10, N = 10):
    '''We must be careful here to create a Numeric array to pass into getValues() to avoid a copy'''
    import Numeric
    from PETSc.MatReuse import MAT_INITIAL_MATRIX

    self.setup()
    A,B = self.createMatrices(m,n,M, N)

    # only use a part of the space allocated for A
    aa = A.getArray()
    a = Numeric.reshape(aa,[M,N])
    a[0,0] = 8
    a[2,2] = 17
    a[0,2] = -3
    A.restoreArray(aa)

    id = Numeric.array([0,2],'i')
    C = A.getSubMatrixRaw(id,id,2,MAT_INITIAL_MATRIX)

    print Numeric.reshape(C.getArray(),[2,2])
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
