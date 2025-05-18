import config.package

class Configure(config.package.PythonPackage):
  def __init__(self, framework):
    config.package.PythonPackage.__init__(self, framework)
    self.pkgname         = 'fenics-ffcx'
    self.buildLanguages  = ['Cxx']
    self.useddirectly    = 0
    return

  def setupDependencies(self, framework):
    config.package.PythonPackage.setupDependencies(self, framework)
    self.basix         = framework.require('config.packages.fenics-basix', self)
    self.ufl           = framework.require('config.packages.fenics-ufl', self)
    self.deps          = [self.basix, self.ufl]
    return
