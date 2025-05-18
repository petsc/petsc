import config.package

class Configure(config.package.PythonPackage):
  def __init__(self, framework):
    config.package.PythonPackage.__init__(self, framework)
    self.pkgname         = 'nanobind'
    self.useddirectly    = 0
