#!/usr/bin/env python
class Ex1:
  '''Saves and loads a matrix in PETSc binary format'''
  def __init__(self):
    return

  def setup(self):
    import PETSc.Base
    import atexit

    PETSc.Base.Base.Initialize()
    atexit.register(PETSc.Base.Base.Finalize)
    return

  def createViewer(self, filename, read = 1):
    from PETSc.PetscViewerFileType import FILE_MODE_READ, FILE_MODE_WRITE
    import PETSc.PetscViewerBinary

    viewer = PETSc.PetscViewerBinary.PetscViewerBinary()
    if read:
      viewer.setFileType(FILE_MODE_READ)
    else:
      viewer.setFileType(FILE_MODE_WRITE)
    viewer.setFilename(filename)
    return viewer

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

  def run(self):
    import PETSc.Mat

    self.setup()
    filename = 'test.bin'
    viewer = self.createViewer(filename, read = 0)
    A = self.createMatrix(10, 10)
    A.view(viewer)
    del viewer
    viewer = self.createViewer(filename)
    B = PETSc.Mat.Mat.load(viewer, 'aij')
    del viewer
    if os.path.isfile(filename):
      os.remove(filename)
    if not A.equal(B):
      raise RuntimeError('Saved matrix was not identical to the original')
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
