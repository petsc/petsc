import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def setOutput(self):
    #self.addDefine('HAVE_TRIANGLE', 0)
    self.addSubstitution('TRIANGLE_INCLUDE', '', 'The Triangle include flags')
    self.addSubstitution('TRIANGLE_LIB', '', 'The Triangle library flags')
    return

  def configure(self):
    self.setOutput()
    return
