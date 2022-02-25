import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit = 'fe4d5f10840c5f62b984364a4d41719f1bc079a2' # master sep-24-2020
    self.download  = ['git://https://github.com/google/googletest.git']
    self.functions = []
    self.includes  = ['gtest/gtest.h']
    self.liblist   = [['libgtest.a']]
    self.pkgname   = 'googletest'
    self.buildLanguages= ['Cxx']
    self.hastests  = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.deps            = [self.mpi]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-Dgtest_build_tests=ON')
    args.append('-Dgmock_build_tests=ON')
    args.append('-Dgtest_build_samples=ON')
    args.append('-DBUILD_GMOCK=ON')
    args.append('-DBUILD_GTEST=OFF')
    if not hasattr(self.compilers, 'CXX'):
      raise RuntimeError("googletest requires a C++ compiler\n")
    return args
