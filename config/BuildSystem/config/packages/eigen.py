import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    import os
    config.package.CMakePackage.__init__(self, framework)
    self.download          = ['hg://https://bitbucket.org/eigen/eigen/','https://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz']
    self.functions         = []
    self.includes          = ['Eigen/Core']
    self.liblist           = []
    self.cxx               = 1
    self.includedir        = os.path.join('include', 'eigen3')
    self.useddirectly      = 0
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.deps          = []
    return

  def formCMakeConfigureArgs(self):
    if not self.cmake.found:
      raise RuntimeError('CMake > 2.5 is needed to build Eigen')

    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DENABLE_OPENMP=OFF')
    return args
