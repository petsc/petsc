import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.compilers = self.framework.require('config.compilers', self)
    self.libraries = self.framework.require('config.libraries', self)
    return

  def checkDot(self):
    '''Verify that the ddot() function is contained in the BLAS library'''
    return self.libraries.check('libblas.a', 'ddot', otherLibs = self.compilers.flibs, fortranMangle = 1)

  def configure(self):
    self.executeTest(self.checkDot)
    return
