import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download  = ['ftp://ftp.mcs.anl.gov/pub/mpi/mpe/mpe2.tar.gz']
    self.functions = ['MPE_Log_event']
    self.includes  = ['mpe.h']
    self.liblist   = [['libmpe.a']]
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi  = framework.require('config.packages.MPI',self)
    self.deps = [self.mpi]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    self.pushLanguage('C')
    args.append('MPI_CFLAGS="'+self.updatePackageCFlags(self.getCompilerFlags())+'"')
    args.append('MPI_CC="'+self.getCompiler()+'"')
    self.popLanguage()

    if hasattr(self.compilers, 'FC'):
      self.pushLanguage('FC')
      args.append('MPI_FFLAGS="'+self.getCompilerFlags()+'"')
      args.append('F77="'+self.getCompiler()+'"')
      args.append('MPI_F77="'+self.getCompiler()+'"')
      self.popLanguage()
    else:
      args.append('--disable-f77')

    args.append('MPI_INC="'+self.headers.toString(self.mpi.include)+'"')
    args.append('MPI_LIBS="'+self.libraries.toStringNoDupes(self.mpi.lib)+'"')
    return args

