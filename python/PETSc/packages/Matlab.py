import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def setOutput(self):
    #self.addDefine('HAVE_MATLAB', 0)
    self.addSubstitution('CMEX', '')
    self.addSubstitution('MCC', '')
    self.addSubstitution('MATLABCOMMAND', '')
    return

  def configure(self):
    self.setOutput()
    return
