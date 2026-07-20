import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version           = '2.0.0'
    self.gitcommit         = self.version
    self.download          = ['git://https://github.com/libxsmm/libxsmm.git','https://github.com/libxsmm/libxsmm/archive/'+self.gitcommit+'.tar.gz']
    self.includes          = ['libxsmm.h']
    self.liblist           = [['libxsmm.a']]
    self.functions         = ['libxsmm_init']
    self.precisions        = ['single', 'double']
    self.complex           = 0
    self.versionname       = 'LIBXSMM_VERSION'
    self.buildLanguages    = ['C', 'Cxx']
    self.minCmakeVersion   = (3,13,0)
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    if not self.compilerFlags.debugging:
      args = self.addArgStartsWith(args, '-DCMAKE_C_FLAGS_RELEASE:STRING', '-DNDEBUG')
    if not hasattr(self.compilers, 'FC'):
      args += ['-DCMAKE_Fortran_COMPILER=NOTFOUND', '-DLIBXSMM_FORTRAN:BOOL=OFF']
    return args
