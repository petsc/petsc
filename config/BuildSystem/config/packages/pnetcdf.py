import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download         = ['http://cucis.ece.northwestern.edu/projects/PnetCDF/Release/parallel-netcdf-1.9.0.tar.gz',
                             'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/parallel-netcdf-1.9.0.tar.gz']
    self.functions        = ['ncmpi_create']
    self.includes         = ['pnetcdf.h']
    self.liblist          = [['libpnetcdf.a']]
    self.downloaddirnames = ['parallel-netcdf-1.9.0']
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.flibs = framework.require('config.packages.flibs',self)
    self.mpi   = framework.require('config.packages.MPI', self)
    self.deps  = [self.mpi,self.flibs]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--with-mpi="'+self.mpi.directory+'"')
    args.append('LIBS="'+self.libraries.toStringNoDupes(self.flibs.lib)+'"')
    return args
