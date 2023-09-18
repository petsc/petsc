import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit              = '1d86d5327a621c030fe61255ce0114116d60d74a' # master 2023-4-2
    self.download               = ['git://https://github.com/Reference-LAPACK/lapack.git','https://github.com/Reference-LAPACK/lapack/archive/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames       = ['netlib-lapack']
    self.includes               = []
    self.liblist                = [['libnlapack.a','libnblas.a']]
    self.precisions             = ['single','double']
    self.functionsFortran       = 1
    self.buildLanguages         = ['FC']
    self.minCmakeVersion        = (2,8,3)
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DLIBRARY_PREFIX=n')
    return args
