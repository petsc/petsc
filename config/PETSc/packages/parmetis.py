import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.download          = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/parmetis-4.0.2-p5.tar.gz']
    self.functions         = ['ParMETIS_V3_PartKway']
    self.includes          = ['parmetis.h']
    self.liblist           = [['libparmetis.a']]
    self.needsMath         = 1

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.sharedLibraries = framework.require('PETSc.utilities.sharedLibraries', self)
    self.scalartypes    = framework.require('PETSc.utilities.scalarTypes',self)
    self.libraryOptions = framework.require('PETSc.utilities.libraryOptions', self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.metis           = framework.require('PETSc.packages.metis', self)
    self.deps            = [self.mpi, self.metis]

  def formCMakeConfigureArgs(self):
    '''Requires the same CMake options as Metis'''
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DGKLIB_PATH=../headers') 
    if self.sharedLibraries.useShared:
      args.append('-DSHARED=1')
    if self.compilerFlags.debugging:
      args.append('-DDEBUG=1')
    if self.libraryOptions.integerSize == 64:
      args.append('-DMETIS_USE_LONGINDEX=1')
    if self.scalartypes.precision == 'double':
      args.append('-DMETIS_USE_DOUBLEPRECISION=1')
    elif self.scalartypes.precision == 'quad':
      raise RuntimeError('METIS cannot be built with quad precision')
    return args
