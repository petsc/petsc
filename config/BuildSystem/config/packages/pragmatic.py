import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    import os
    config.package.CMakePackage.__init__(self, framework)
    self.download          = ['git://https://github.com/meshadaptation/pragmatic.git']
    self.gitcommit         = '134468543d3c6052803c69fca3a714f6e2ce0275'
    self.functions         = ['pragmatic_2d_init']
    self.includes          = ['pragmatic.h']
    self.liblist           = [['libpragmatic.a']]
    self.needsMath         = 1
    self.includedir        = os.path.join('include', 'pragmatic')
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.scalartypes     = framework.require('PETSc.options.scalarTypes',self)
    self.indexTypes      = framework.require('PETSc.options.indexTypes', self)
    self.metis           = framework.require('config.packages.metis', self)
    self.deps            = [self.metis]
    return

  def formCMakeConfigureArgs(self):
    if not self.cmake.found:
      raise RuntimeError('CMake > 2.5 is needed to build Pragmatic')

    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DMETIS_DIR='+self.metis.getInstallDir())
    args.append('-DENABLE_VTK=OFF')
    args.append('-DENABLE_OPENMP=OFF')
    if self.checkSharedLibrariesEnabled():
      args.append('-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON')
    if self.indexTypes.integerSize == 64:
      raise RuntimeError('Pragmatic cannot be built with 64-bit integers')
    if self.scalartypes.precision == 'single':
      raise RuntimeError('Pragmatic cannot be built with single precision')
    elif self.scalartypes.precision == 'quad':
      raise RuntimeError('Pragmatic cannot be built with quad precision')
    return args
