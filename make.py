#!/usr/bin/env python
import user
import builder
import script

import os

class Make(script.Script):
  def __init__(self, builder):
    import RDict
    import sys

    script.Script.__init__(self, sys.argv[1:], RDict.RDict())
    self.builder = builder
    self.builder.pushLanguage('C')
    return

  def setupHelp(self, help):
    import nargs

    help = script.Script.setupHelp(self, help)
    help.addArgument('PETSc', '-build-shared-libraries=<bool>', nargs.ArgBool(None, 0, 'Build the PETSc shared libraries', isTemporary = 1))
    return help

  def setup(self):
    script.Script.setup(self)
    self.builder.setup()
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
  petscArch = property(getPetscArch, setPetscArch, doc = 'The root of the PETSc tree')

  def configure(self):
    return

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

    libDir = os.path.join(self.petscDir, 'lib', 'lib'+self.argDB['BOPT'], self.petscArch)
    tmpDir = os.path.join(libDir, 'tmp')
    self.logPrint('Starting shared library build into '+libDir, debugSection = 'build')
    builder.pushConfiguration('Shared Libraries')
    builder.setLinkerExtraArguments(' '.join(self.getExternalLibraries()+[self.argDB['BLASLAPACK_LIB'], self.argDB['MPI_LIB'], self.argDB['LIBS'], self.argDB['FLIBS']]))
    if os.path.exists(tmpDir):
      shutil.rmtree(tmpDir)
    os.mkdir(tmpDir)
    for libBase in ['libpetsc', 'libpetscvec', 'libpetscmat', 'libpetscdm', 'libpetscksp', 'libpetscsnes', 'libpetscts']:
      library       = os.path.join(libDir, libBase+'.'+self.argDB['LIB_SUFFIX'])
      if not os.path.isfile(library):
        library     = os.path.join(libDir, 'lt_'+libBase+'.'+self.argDB['LIB_SUFFIX'])
      sharedLibrary = os.path.join(libDir, libBase+'.so')
      # Check archive against ${INSTALL_LIB_DIR}/$$LIBNAME.${SLSUFFIX}
      self.logPrint('Building shared library: '+sharedLibrary, debugSection = 'build')
      self.builder.link(self.expandArchive(library, tmpDir), os.path.join('lib', 'triangle.so'), shared = 1)
      [os.remove(os.path.join(tmpDir, f)) for f in os.listdir(tmpDir)]
    shutil.rmtree(tmpDir)
    builder.popConfiguration()
    self.logPrint('Ended shared library build', debugSection = 'build')
    return

  def run(self):
    self.setup()
    self.logPrint('Starting', debugSection = 'build')
    self.configure()
    self.buildSharedLibraries(self.builder)
    self.logPrint('Done', debugSection = 'build')
    return 1

if __name__ == '__main__':
  Make(builder.Builder()).run()
