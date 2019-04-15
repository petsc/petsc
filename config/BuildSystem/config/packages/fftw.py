import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    # host locally as fftw.org url can expire after new release.
    self.download   = ['http://www.fftw.org/fftw-3.3.8.tar.gz',
                       'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/fftw-3.3.8.tar.gz']
    self.functions  = ['fftw_malloc','fftw_mpi_cleanup']
    self.includes   = ['fftw3-mpi.h']
    self.liblist    = [['libfftw3_mpi.a','libfftw3.a']]
    self.pkgname    = 'fftw3'
    self.precisions = ['double']
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi  = framework.require('config.packages.MPI',self)
    self.blasLapack = self.framework.require('config.packages.BlasLapack',self)
    self.deps = [self.mpi,self.blasLapack]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    self.framework.pushLanguage('C')
    args.append('MPICC="'+self.framework.getCompiler()+'"')
    self.framework.popLanguage()
    args.append('--enable-mpi')
    if self.mpi.lib:
      args.append('LIBS="'+self.libraries.toStringNoDupes(self.mpi.lib)+'"')
    return args
