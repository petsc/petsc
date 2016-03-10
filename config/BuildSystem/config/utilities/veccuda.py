import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.cuda = framework.require('config.packages.cuda', self)
    self.cusp = framework.require('config.packages.cusp', self)
    self.viennacl = framework.require('config.packages.viennacl', self)
    return

  def configureVecCUDA(self):
    '''Configure VecCUDA as fallback CUDA vector if CUSP and VIENNACL are not present'''
    if self.cuda.found and not self.cusp.found and not self.viennacl.found:
      self.addDefine('HAVE_VECCUDA','1')

  def configure(self):
    self.executeTest(self.configureVecCUDA)
    return
