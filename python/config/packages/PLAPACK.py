import configure

class Configure(configure.Configure):
  def __init__(self, framework):
    configure.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def setOutput(self):
    #self.addDefine('HAVE_PLAPACK', 0)
    self.addSubstitution('PLAPACK_INCLUDE', '', 'The PLAPACK include flags')
    self.addSubstitution('PLAPACK_LIB', '', 'The PLAPACK library flags')
    return

  def configure(self):
    self.setOutput()
    return
