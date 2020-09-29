import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version          = '2.7.2'
    self.gitcommit        = 'ac89ea4' # master sep-26-2020
    self.download         = ['git://https://github.com/coin-or/ADOL-C.git','https://www.coin-or.org/download/source/ADOL-C/ADOL-C-' + self.version + '.tgz']
    self.includes         = ['adolc/adolc.h']
    self.liblist          = [['libadolc.a']]
    self.functions        = ['myalloc2','myfree2']
    self.cxx              = 1
    self.requirescxx11    = 1
    self.precisions       = ['double']
    self.complex          = 0
    self.downloaddirnames = ['ADOL-C']
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.colpack = framework.require('config.packages.ColPack', self)
    self.deps    = [self.colpack]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--without-boost')
    args.append('--enable-sparse')
    args.append('--with-colpack="'+self.colpack.directory+'"')
    return args
