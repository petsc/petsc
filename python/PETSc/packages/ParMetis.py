import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def configureLibrary(self):
    '''Find a ParMetis installation and check if it can work with PETSc'''
    return

  def setOutput(self):
    #self.addDefine('HAVE_PARMETIS', 0)
    self.addSubstitution('PARMETIS_INCLUDE', '')
    self.addSubstitution('PARMETIS_LIB', '')
    return

  def configure(self):
    self.executeTest(self.configureLibrary)
    self.setOutput()
    return
