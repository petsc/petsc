import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def setOutput(self):
    #self.addDefine('HAVE_PARMETIS', 0)
    self.addSubstitution('PARMETIS_INCLUDE', '', 'The ParMetis include flags')
    self.addSubstitution('PARMETIS_LIB', '', 'The ParMetis library flags')
    return

  def configure(self):
    self.setOutput()
    return
