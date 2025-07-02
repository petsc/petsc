import config.package

class Configure(config.package.PythonPackage):
  def __init__(self, framework):
    config.package.PythonPackage.__init__(self, framework)
    self.version         = "2024.2.0"
    self.pkgname         = 'fenics_ufl'
    self.buildLanguages  = ['Cxx']
    self.useddirectly    = 0

  def setupDependencies(self, framework):
    config.package.PythonPackage.setupDependencies(self, framework)
    self.pathspec      = framework.require('config.packages.pathspec', self)
    self.deps          = [self.pathspec]
    return
