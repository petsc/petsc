import configure

class Configure(configure.Configure):
  def __init__(self, framework):
    configure.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def setOutput(self):
    #self.addDefine('HAVE_ADIC', 0)
    #self.addDefine('HAVE_ADIFOR', 0)
    self.addSubstitution('ADIC_DEFINES', '', '???')
    self.addSubstitution('ADIC_CC', '', '???')
    return

  def configure(self):
    self.setOutput()
    return
