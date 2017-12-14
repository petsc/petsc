import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download        = ['ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.5.0.tar.gz']
    self.functions       = ['nccreate']
    self.includes        = ['netcdf.h']
    self.liblist         = [['libnetcdf.a']]
    self.cxx             = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi     = framework.require('config.packages.MPI', self)
    self.pnetcdf = framework.require('config.packages.pnetcdf', self)
    self.hdf5    = framework.require('config.packages.hdf5', self)
    self.deps    = [self.mpi, self.hdf5]
    self.odeps   = [self.pnetcdf]
    return

  def formGNUConfigureArgs(self):
    ''' disable DAP and HDF4, enable NetCDF4'''
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('CPPFLAGS="'+self.headers.toString(self.hdf5.include)+'"')
    args.append('LIBS="'+self.libraries.toString(self.hdf5.dlib)+' '+self.compilers.LIBS+'"')
    args.append('--enable-netcdf-4')
    if self.pnetcdf.found:
      args.append('--enable-pnetcdf')
    args.append('--disable-dap')
    args.append('--disable-dynamic-loading') #This was disabled in v4.3.2 - but enabled in subsequent versions - giving config errors on freebsd (wrt -ldl)
    args.append('--disable-hdf4')
    return args
