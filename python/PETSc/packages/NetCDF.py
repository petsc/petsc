import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers', self)
    return

  def configureLibrary(self):
    '''Find a NetCDF installation and check if it can work with PETSc'''
    return

  def setOutput(self):
    #self.addDefine('HAVE_TRIANGLE', 0)
    self.addSubstitution('NETCDF_INCLUDE', '')
    self.addSubstitution('NETCDF_LIB', '')
    return

  def configure(self):
    self.executeTest(self.configureLibrary)
    self.setOutput()
    return
