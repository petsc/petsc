import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit     = 'master'
    self.download      = ['git://https://github.com/CSCsw/ColPack.git']
    self.includes      = ['ColPack/ColPackHeaders.h']
    self.liblist       = [['libColPack.a']]
    self.functionsCxx  = [1,'void current_time();','current_time()']
    self.requirescxx11 = 1
    self.cxx           = 1
    self.precisions    = ['double']
    self.complex       = 0
    self.builddir      = 'yes'
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.openmp = framework.require('config.packages.openmp',self)
    self.deps = [self.openmp]
    return

  def formGNUConfigureArgs(self):
    import os
    self.packageDir = os.path.join(self.packageDir,'build','automake')
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    # args.append('--enable-examples=no')  #  this option doesn't work to prevent processing examples
    # args.append('--enable-openmp=no')    # this option doesn't work, the OpenMP is hardwired in the source code
    return args
