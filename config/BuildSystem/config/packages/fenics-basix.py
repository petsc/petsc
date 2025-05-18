import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version         = '0.9.0'
    self.gitcommit       = 'v'+self.version
    self.download        = ['git://https://github.com/FEniCS/basix/']
    self.functions       = []
    self.includes        = ['basix/finite-element.h']
    self.liblist         = [['libbasix.a']]
    self.buildLanguages  = ['Cxx']
    self.pkgname         = 'basix'
    self.cmakelistsdir   = 'cpp'
    self.useddirectly    = 0
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    return args
