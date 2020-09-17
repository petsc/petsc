import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit = 'master'
    self.download  = ['git://https://github.com/google/googletest.git']
    self.functions = []
    self.includes  = ['gtest/gtest.h']
    self.liblist   = [['libgtest.a']]
    self.pkgname   = 'googletest'
    self.cxx       = 1
    self.hastests  = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.deps            = [self.mpi]
    return

  def formCMakeConfigureArgs(self):
    if not self.cmake.found:
      raise RuntimeError('CMake > 2.5 is needed to build googletest\nSuggest adding --download-cmake to ./configure arguments')

    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-Dgtest_build_tests=ON')
    args.append('-Dgmock_build_tests=ON')
    args.append('-Dgtest_build_samples=ON')
    args.append('-DBUILD_GMOCK=ON')
    args.append('-DBUILD_GTEST=OFF')
    if hasattr(self.compilers, 'CXX'):
      self.pushLanguage('Cxx')
      args.append('-DMPI_CXX_COMPILER="'+self.getCompiler()+'"')
      args.append('-DCMAKE_CXX_FLAGS:STRING="'+self.updatePackageCxxFlags(self.getCompilerFlags())+'"')
    else:
        raise RuntimeError("googletest requires a C++ compiler\n")
    self.popLanguage()
    self.pushLanguage('C')
    args.append('-DMPI_C_COMPILER="'+self.getCompiler()+'"')
    args.append('-DCMAKE_C_FLAGS:STRING="'+self.updatePackageCFlags(self.getCompilerFlags())+'"')
    self.popLanguage()
    return args
