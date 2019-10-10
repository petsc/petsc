import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version          = '1.11.2'
    self.versionname      = 'PNETCDF_VERSION'
    self.download         = ['https://parallel-netcdf.github.io/Release/pnetcdf-'+self.version+'.tar.gz',
                             'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/pnetcdf-'+self.version+'.tar.gz']
    self.functions        = ['ncmpi_create']
    self.includes         = ['pnetcdf.h']
    self.liblist          = [['libpnetcdf.a']]
    self.useddirectly     = 0
    self.installwithbatch = 0
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.flibs = framework.require('config.packages.flibs',self)
    self.mpi   = framework.require('config.packages.MPI', self)
    self.deps  = [self.mpi,self.flibs]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    self.addToArgs(args,'LIBS',self.libraries.toStringNoDupes(self.flibs.lib))
    return args
