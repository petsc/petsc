import configure

class Configure(configure.Configure):
  def __init__(self, framework):
    configure.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def setOutput(self):
    #self.addDefine('HAVE_MATLAB', 0)
    self.addSubstitution('CMEX', '', 'The Matlab MEX compiler')
    self.addSubstitution('MCC', '', 'The Matlab C compiler')
    self.addSubstitution('MATLABCOMMAND', '', 'The Matlab executable')
    return

  def configure(self):
    self.setOutput()
    return
