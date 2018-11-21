import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
#    self.version          = '2.6.0'
    self.download         = ['https://www.coin-or.org/download/source/ADOL-C/ADOL-C-2.6.0.tgz']
    self.includes         = ['adolc/adolc.h']
    self.liblist          = [['libadolc.la']]
    self.functions        = []
    self.cxx              = 1
#    self.requirescxx11    = 1
#    self.precisions       = ['double']
#    self.complex          = 0
    self.downloaddirnames = ['ADOL-C-2.6.0']
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
#    self.colpack = framework.require('config.packages.ColPack',self) # TODO
#    self.deps    = [self.colpack] # TODO
    self.deps = []
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--enable-docexa')
#    args.append('--enable-sparse')
#    args.append('--with-colpack-dir=$COLPACK_HOME') # TODO
    return args
