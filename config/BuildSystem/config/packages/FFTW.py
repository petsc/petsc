import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    # host locally as fftw.org url can expire after new release.
    self.version    = '3.3.11'
    self.download   = ['https://www.fftw.org/fftw-'+self.version+'.tar.gz',
                       'https://web.cels.anl.gov/projects/petsc/download/externalpackages/fftw-'+self.version+'.tar.gz']
    self.functions  = ['fftw_malloc']
    self.includes   = ['fftw3.h']
    self.liblist    = [['libfftw3_mpi.a','libfftw3.a'],['libfftw3.a']]
    self.precisions = ['double']
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi  = framework.require('config.packages.MPI',self)
    self.blasLapack = self.framework.require('config.packages.BlasLapack',self)
    self.deps = [self.blasLapack]
    self.odeps = [self.mpi]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    self.pushLanguage('C')
    self.popLanguage()
    if self.mpi.found and not self.mpi.usingMPIUni:
      args.append('MPICC="'+self.getCompiler()+'"')
      args.append('--enable-mpi')
      if self.mpi.lib:
        args.append('LIBS="'+self.libraries.toStringNoDupes(self.mpi.lib)+'"')
    return args
