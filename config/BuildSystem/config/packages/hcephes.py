import config.package


class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version                = '0.4.1'
    self.gitcommit              = '0.4.1'
    self.download               = ['git://https://github.com/limix/hcephes.git','https://github.com/limix/hcephes/archive/refs/tags/'+self.gitcommit+'.tar.gz']
    self.includes               = ['hcephes.h']
    self.hastests               = 1
    self.functions              = ['hcephes_erf']
    self.liblist                = [['libhcephes.a']]
    self.precisions             = ['double']
    self.useddirectly           = 0
  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.mathlib       = framework.require('config.packages.mathlib', self)
    self.deps          = [self.mathlib]
