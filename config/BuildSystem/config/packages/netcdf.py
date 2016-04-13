import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download  = ['ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.3.2.tar.gz',
                      'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/netcdf-4.3.2.tar.gz']
    self.functions       = ['nccreate']
    self.includes        = ['netcdf.h']
    self.liblist         = [['libnetcdf.a']]
    self.cxx             = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi   = framework.require('config.packages.MPI', self)
    self.hdf5  = framework.require('config.packages.hdf5', self)
    self.deps  = [self.mpi, self.hdf5]
    return

  def formGNUConfigureArgs(self):
    ''' disable DAP and HDF4, enable NetCDF4'''
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('CPPFLAGS="'+self.headers.toString(self.hdf5.include)+'"')
    args.append('LIBS="'+self.libraries.toString(self.hdf5.dlib)+' '+self.compilers.LIBS+'"')
    args.append('--enable-netcdf-4')
    args.append('--disable-dap')
    args.append('--disable-hdf4')
    return args
