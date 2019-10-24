import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.lookforbydefault  = 1
    return

  def __str__(self):
    return ''

  def setupHelp(self,help):
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.libraries       = framework.require('config.libraries', self)
    return

  def configure(self):
    self.framework.packages.append(self)
    self.lib   = self.libraries.math
    self.dlib  = self.libraries.math
    self.found = 1
    return
