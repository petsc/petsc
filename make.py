#!/usr/bin/env python
import user
import maker
import script

import os

class Make(maker.Make):
  def __init__(self, builder = None):
    maker.Make.__init__(self, builder)
    self.libBases = ['libpetsc', 'libpetscvec', 'libpetscmat', 'libpetscdm', 'libpetscksp', 'libpetscsnes', 'libpetscts']
    return

  def setupHelp(self, help):
    import nargs

    help = maker.Make.setupHelp(self, help)
    help.addArgument('PETSc', '-build-shared-libraries=<bool>', nargs.ArgBool(None, 0, 'Build the PETSc shared libraries', isTemporary = 1))
    return help

  def setupDependencies(self, sourceDB):
    for libBase in self.libBases:
      sourceDB.addDependency(os.path.join(self.libDir, 'test_'+libBase+'.so'), os.path.join(self.libDir, libBase+'.'+self.argDB['LIB_SUFFIX']))
    return

  def setup(self):
    maker.Make.setup(self)
    if not self.argDB['build-shared-libraries']:
      import sys

      sys.exit('******************************** INVALID COMMAND ********************************\nThis is an alpha build system, and currently can only build the shared libraries.')
    return

  def getPetscDir(self):
    if not hasattr(self, '_petscDir'):
      if 'PETSC_DIR' in os.environ:
        self._petscDir = os.environ['PETSC_DIR']
      elif 'PETSC_DIR' in self.argDB:
        self._petscDir = self.argDB['PETSC_DIR']
      else:
        self._petscDir = os.getcwd()
    return self._petscDir
  def setPetscDir(self, petscDir):
    self._petscDir = petscDir
  petscDir = property(getPetscDir, setPetscDir, doc = 'The root of the PETSc tree')

  def getPetscArch(self):
    if not hasattr(self, '_petscArch'):
      if 'PETSC_DIR' in os.environ:
        self._petscArch = os.environ['PETSC_ARCH']
      elif 'PETSC_DIR' in self.argDB:
        self._petscArch = self.argDB['PETSC_ARCH']
      else:
        self._petscArch = os.getcwd()
    return self._petscArch
  def setPetscArch(self, petscArch):
    self._petscArch = petscArch
  petscArch = property(getPetscArch, setPetscArch, doc = 'The PETSc build configuration')

  def getLibDir(self):
    if not hasattr(self, '_libDir'):
      self._libDir = os.path.join(self.petscDir, 'lib',  self.petscArch)
    return self._libDir
  def setLibDir(self, libDir):
    self._libDir = libDir
  libDir = property(getLibDir, setLibDir, doc = 'The PETSc library directory')

  def expandArchive(self, archive, dir):
    self.executeShellCommand(' '.join(['cd', dir, ';', self.argDB['AR'], 'x', archive]))
    return [os.path.join(dir, f) for f in os.listdir(dir)]

  def getExternalLibraries(self):
    nonconfigurePackages = ['AMS', 'SPAI', 'LUSOL', 'RAMG', 'TAU', 'ADIFOR', 'TRILINOS', 'HYPRE', 'SAMG', 'PNETCDF', 'HDF4', 'CHACO', 'JOSTLE', 'PARTY', 'SCOTCH']
    for package in nonconfigurePackages:
      if not package+'_LIB' in self.argDB:
        self.argDB[package+'_LIB'] = ''
    packages = ['MPE', 'BLOCKSOLVE', 'PVODE', 'PARMETIS', 'AMS', 'SPAI', 'X11', 'MATLAB', 'LUSOL', 'DSCPACK', 'RAMG',
                'TAU', 'ADIFOR', 'SUPERLU_DIST', 'SUPERLU', 'SPOOLES', 'UMFPACK', 'TRILINOS', 'HYPRE', 'MUMPS',
                'MATHEMATICA', 'TRIANGLE', 'PLAPACK', 'SAMG', 'PNETCDF', 'HDF4', 'CHACO', 'JOSTLE', 'PARTY', 'SCOTCH']
    return [self.argDB[package+'_LIB'] for package in packages]

  def buildSharedLibraries(self, builder):
    import shutil

    tmpDir = os.path.join(self.libDir, 'tmp')
    self.logPrint('Starting shared library build into '+self.libDir, debugSection = 'build')
    builder.pushConfiguration('Shared Libraries')
    builder.setLinkerExtraArguments(' '.join(self.getExternalLibraries()+[self.argDB['BLASLAPACK_LIB'], self.argDB['MPI_LIB'], self.argDB['LIBS'], self.argDB['FLIBS']]))
    if os.path.exists(tmpDir):
      shutil.rmtree(tmpDir)
    os.mkdir(tmpDir)
    for libBase in self.libBases:
      library       = os.path.join(self.libDir, libBase+'.'+self.argDB['LIB_SUFFIX'])
      sharedLibrary = os.path.join(self.libDir, 'test_'+libBase+'.so')
      self.logPrint('Building shared library: '+sharedLibrary, debugSection = 'build')
      if builder.shouldCompile([library], sharedLibrary):
        builder.link(self.expandArchive(library, tmpDir), sharedLibrary, shared = 1)
        [os.remove(os.path.join(tmpDir, f)) for f in os.listdir(tmpDir)]
      builder.shouldCompile.sourceDB.updateSource(library)
    shutil.rmtree(tmpDir)
    builder.popConfiguration()
    self.logPrint('Ended shared library build', debugSection = 'build')
    return

  def build(self, builder):
    self.buildSharedLibraries(builder)
    return

if __name__ == '__main__':
  Make().run()
