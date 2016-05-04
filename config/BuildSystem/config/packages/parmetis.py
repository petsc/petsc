import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = 'v4.0.3-p3'
    self.download          = ['git://https://bitbucket.org/petsc/pkg-parmetis.git','http://ftp.mcs.anl.gov/pub/petsc/externalpackages/parmetis-4.0.3-p3.tar.gz']
    self.functions         = ['ParMETIS_V3_PartKway']
    self.includes          = ['parmetis.h']
    self.liblist           = [['libparmetis.a']]
    self.needsMath         = 1
    self.hastests          = 1

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.scalartypes    = framework.require('PETSc.options.scalarTypes',self)
    self.indexTypes     = framework.require('PETSc.options.indexTypes', self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.metis           = framework.require('config.packages.metis', self)
    self.deps            = [self.mpi, self.metis]

  def formCMakeConfigureArgs(self):
    '''Requires the same CMake options as Metis'''
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DGKLIB_PATH=../headers')
    if self.mpi.include:
      args.append('-DMPI_INCLUDE_PATH='+self.mpi.include[0])
    if self.checkSharedLibrariesEnabled():
      args.append('-DSHARED=1')
      args.append('-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON')
    if self.compilerFlags.debugging:
      args.append('-DDEBUG=1')
    if self.indexTypes.integerSize == 64:
      args.append('-DMETIS_USE_LONGINDEX=1')
    if self.scalartypes.precision == 'double':
      args.append('-DMETIS_USE_DOUBLEPRECISION=1')
    elif self.scalartypes.precision == 'quad':
      raise RuntimeError('METIS cannot be built with quad precision')
    return args

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    if self.libraries.check(self.lib, 'ParMETIS_ComputeVertexSeparator',otherLibs=self.metis.lib+self.mpi.lib+self.libraries.math):
      self.ComputeVertexSeparator = 1
    else:
      self.ComputeVertexSeparator = 0
    return


