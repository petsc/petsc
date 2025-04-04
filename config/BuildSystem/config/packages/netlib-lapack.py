import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit              = 'v3.12.0'
    self.download               = ['git://https://github.com/Reference-LAPACK/lapack.git','https://github.com/Reference-LAPACK/lapack/archive/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames       = ['netlib-lapack','lapack']
    self.includes               = []
    self.liblist                = [['libnlapack.a','libnblas.a']]
    self.precisions             = ['single','double']
    self.functionsFortran       = 1
    self.buildLanguages         = ['FC']
    self.minCmakeVersion        = (2,8,3)
    self.cinterface             = False
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument(self.PACKAGE,'-with-netlib-lapack-c-bindings=<bool>',nargs.ArgBool(None,0,'Use/build the C interface (CBLAS and LAPACKE) for '+self.name+' (PETSc does not need it)'))
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    if not self.cinterface:
      # use a prefix to avoid conflict with another LAPACK already installed in the system
      args.append('-DLIBRARY_PREFIX=n')
    else:
      # build the C interface; in this case we do not use a prefix because PaStiX build
      # system does not support it (requires CBLAS/LAPACKE)
      args.append('-DCBLAS:BOOL=ON')
      args.append('-DLAPACKE:BOOL=ON')
    return args

  def generateLibList(self, framework):
    if self.cinterface:
      self.liblist = [['liblapacke.a','libcblas.a','liblapack.a','libblas.a']]
    return config.package.Package.generateLibList(self, framework)

  def configureLibrary(self):
    self.cinterface = self.argDB['with-netlib-lapack-c-bindings']
    config.package.Package.configureLibrary(self)
