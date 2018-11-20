import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version    = 'ADOL-C-2.6.0.tgz'
    self.download   = ['https://www.coin-or.org/download/source/ADOL-C/'+self.version+'.tar.gz']
    self.functions  = []
    self.includes   = ['adolc.h']
    self.liblist    = [['libadolc.la']]
    self.cxx        = 1
    self.precisions = ['double']
    self.complex    = 0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.ADOLC = framework.require('config.packages.ADOLC', self)
    self.deps  = [self.ADOLC]
    return

  def Install(self):
    import os

    self.framework.pushLanguage('Cxx')

    makeam = os.path.join(self.packageDir, 'Makefile.am')

    g = open(makeam, 'w')
    g
#TODO
