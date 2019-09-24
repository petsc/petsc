import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    return

  def __str__(self):
    return ''

  def setupHelp(self,help):
    return

  def configure(self):
    self.lib   = self.compilers.cxxlibs
    self.dlib  = self.compilers.cxxlibs
    self.found = 1
    return
