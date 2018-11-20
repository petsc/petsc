import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version          = 'ADOL-C-2.6.0'
    self.download         = ['https://www.coin-or.org/download/source/ADOL-C/'+self.version+'.tar.gz']
    self.includes         = ['adolc.h']
    self.liblist          = [['libadolc.la']]
    self.functions        = []  # TODO: Include function for testing
    self.lookforbydefault = 0
    self.cxx              = 1
    self.precisions       = ['double']
    self.complex          = 0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--with-docexa')
    return args
