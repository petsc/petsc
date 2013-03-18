import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.downloadpath    = 'http://www.unidata.ucar.edu/downloads/netcdf/ftp/'
    self.downloadext     = 'tar.gz'
    self.downloadversion = '4.2.1.1'
    self.functions       = ['nccreate']
    self.includes        = ['netcdf.h']
    self.liblist         = [['libnetcdf.a']]
    self.cxx             = 1
    self.excludedDirs    = ['netcdf-cxx']
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi   = framework.require('config.packages.MPI', self)
    self.hdf5  = framework.require('config.packages.hdf5', self)
    self.odeps = [self.mpi, self.hdf5]
    self.deps  = [self.mpi]
    return

  def formGNUConfigureExtraArgs(self):
    '''Specify archiver, libdir, disable DAP and HDF4, enable NetCDF4'''
    args = []
    args.append('AR="'+self.setCompilers.AR+'"')
    args.append('ARFLAGS="'+self.setCompilers.AR_FLAGS+'"')
    args.append('CPPFLAGS="'+self.headers.toString(self.hdf5.include)+'"')
    args.append('LIBS="'+self.libraries.toString(self.hdf5.lib)+'"')
    args.append('--libdir='+os.path.join(self.installDir,self.libdir))
    args.append('--disable-dap')
    args.append('--disable-hdf4')
    args.append('--enable-netcdf-4')
    return args
