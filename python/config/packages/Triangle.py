import configure

class Configure(configure.Configure):
  def __init__(self, framework):
    configure.Configure.__init__(self, framework)
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
