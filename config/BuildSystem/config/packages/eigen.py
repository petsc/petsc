import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    import os
    config.package.CMakePackage.__init__(self, framework)
    self.version       = '3.3.7'
    self.gitcommit     = self.version
    self.download      = ['git://https://gitlab.com/libeigen/eigen','https://gitlab.com/libeigen/eigen/-/archive/'+self.gitcommit+'/eigen-'+self.gitcommit+'.tar.gz']
    self.functions     = []
    self.includes      = ['Eigen/Core']
    self.liblist       = []
    self.cxx           = 1
    self.pkgname       = 'eigen3'
    self.includedir    = os.path.join('include', 'eigen3')
    self.useddirectly  = 0
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.deps          = []
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DENABLE_OPENMP=OFF')
    return args
