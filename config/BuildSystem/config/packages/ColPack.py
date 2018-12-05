import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit     = 'master'
    self.download      = ['git://https://github.com/CSCsw/ColPack.git']
    self.includes      = ['ColPack/ColPackHeaders.h']
    self.liblist       = [['libColPack.a']]
    self.functionsCxx  = [1,'void current_time();','current_time()']
    self.requirescxx11 = 1
    self.cxx           = 1
    self.precisions    = ['double']
    self.complex       = 0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.deps = []
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
#    args.append('--disable-openmp')
    return args
