import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version          = '2.6.0'
    self.download         = ['https://www.coin-or.org/download/source/ADOL-C/ADOL-C-'+self.version+'.tgz']
    self.includes         = ['adolc.h', 'adalloc.h']
    self.liblist          = [['libadolc.la']]
    self.functions        = ['myalloc2']
    self.lookforbydefault = 0
    self.cxx              = 1
    self.requirescxx11    = 1
    self.precisions       = ['double']
    self.complex          = 0
#   self.name             = 'ADOL-C'
#   self.package          = 'adolc'
#   self.PACKAGE          = 'ADOLC'
#   self.downloadname     = 'ADOL-C'
    self.downloaddirnames = ['ADOL-C-'+self.version]
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
#    self.colpack = framework.require('config.packages.ColPack',self) # TODO
#    self.deps    = [self.colpack] # TODO
    return

#  def Install(self):
#    print('Hello world!')
#    return self.installDir

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--enable-sparse')
#    args.append('--with-colpack-dir=$COLPACK_HOME') # TODO
    return args
