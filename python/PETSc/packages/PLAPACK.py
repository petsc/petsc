import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def setOutput(self):
    #self.addDefine('HAVE_PLAPACK', 0)
    self.addSubstitution('PLAPACK_INCLUDE', '')
    self.addSubstitution('PLAPACK_LIB', '')
    return

  def configure(self):
    self.setOutput()
    return
