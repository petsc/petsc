import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers', self)
    self.downloadname = 'Pvode'
    return

  def configureLibrary(self):
    '''Find a PVODE installation and check if it can work with PETSc'''
#    self.framework.packages.append(self)
    return



    return

  def configure(self):
    self.executeTest(self.configureLibrary)
    return
