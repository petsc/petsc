import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version          = '4.0.3'
    self.versionname      = 'PARMETIS_MAJOR_VERSION.PARMETIS_MINOR_VERSION.PARMETIS_SUBMINOR_VERSION'
    self.gitcommit         = 'v'+self.version+'-p9'
    self.download          = ['git://https://bitbucket.org/petsc/pkg-parmetis.git','https://bitbucket.org/petsc/pkg-parmetis/get/'+self.gitcommit+'.tar.gz']
    self.functions         = ['ParMETIS_V3_PartKway']
    self.includes          = ['parmetis.h']
    self.liblist           = [['libparmetis.a']]
    self.hastests          = 1
    self.downloaddirnames  = ['petsc-pkg-parmetis']
    self.downloadonWindows = 1

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.mpi           = framework.require('config.packages.MPI',self)
    self.metis         = framework.require('config.packages.metis', self)
    self.mathlib       = framework.require('config.packages.mathlib',self)
    self.deps          = [self.mpi, self.metis, self.mathlib]

  def formCMakeConfigureArgs(self):
    '''Requires the same CMake options as Metis'''
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DGKLIB_PATH=../headers')
    args.append('-DMETIS_PATH='+self.metis.directory)
    if self.mpi.include:
      args.append('-DMPI_INCLUDE_PATH="'+self.mpi.include[0]+'"')
    if not config.setCompilers.Configure.isWindows(self.setCompilers.CC, self.log) and self.checkSharedLibrariesEnabled():
      args.append('-DSHARED=1')
      args.append('-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON')
    if self.compilerFlags.debugging:
      args.append('-DDEBUG=1')
    if config.setCompilers.Configure.isWindows(self.setCompilers.CC, self.log):
      args.append('-DMSVC=1')
    if self.getDefaultIndexSize() == 64:
      args.append('-DMETIS_USE_LONGINDEX=1')
    return args

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    if self.libraries.check(self.lib, 'ParMETIS_ComputeVertexSeparator',otherLibs=self.metis.lib+self.mpi.lib+self.mathlib.lib):
      self.ComputeVertexSeparator = 1
    else:
      self.ComputeVertexSeparator = 0
    return


