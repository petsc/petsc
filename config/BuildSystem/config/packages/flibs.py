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

  def configure(self):
    self.lib   = self.compilers.flibs
    self.dlib  = self.compilers.flibs
    self.found = 1
    return
