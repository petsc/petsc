import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    import os
    config.package.CMakePackage.__init__(self, framework)
    self.download          = ['GitOnly']
    self.gitcommit         = '02d86b0974c2ce45fa6dc7b709c8f25ef5ef0112'
    self.giturls           = ['https://github.com/pefarrell/pragmatic.git']
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
    return

  def formCMakeConfigureArgs(self):
    if not self.cmake.found:
      raise RuntimeError('CMake > 2.5 is needed to build Pragmatic')

    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DMETIS_DIR='+self.metis.getInstallDir())
    args.append('-DPRAGMATIC_ENABLE_VTK=OFF')
    if self.indexTypes.integerSize == 64:
      raise RuntimeError('Pragmatic cannot be built with 64-bit integers')
    if self.scalartypes.precision == 'single':
      raise RuntimeError('Pragmatic cannot be built with single precision')
    elif self.scalartypes.precision == 'quad':
      raise RuntimeError('Pragmatic cannot be built with quad precision')
    return args
