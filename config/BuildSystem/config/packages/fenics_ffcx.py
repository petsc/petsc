import config.package

class Configure(config.package.PythonPackage):
  def __init__(self, framework):
    config.package.PythonPackage.__init__(self, framework)
    self.pkgname         = 'fenics_ffcx'
    self.buildLanguages  = ['Cxx']
    self.useddirectly    = 0
    return

  def setupDependencies(self, framework):
    config.package.PythonPackage.setupDependencies(self, framework)
    self.cffi          = framework.require('config.packages.cffi', self)
    self.basix         = framework.require('config.packages.fenics-basix', self)
    self.ufl           = framework.require('config.packages.fenics-ufl', self)
    self.deps          = [self.cffi, self.basix, self.ufl]
    return
