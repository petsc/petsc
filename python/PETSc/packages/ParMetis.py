import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def setOutput(self):
    #self.addDefine('HAVE_PARMETIS', 0)
    self.addSubstitution('PARMETIS_INCLUDE', '')
    self.addSubstitution('PARMETIS_LIB', '')
    return

  def configure(self):
    self.setOutput()
    return
