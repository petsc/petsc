import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.downloadpath     = 'http://www.unidata.ucar.edu/downloads/netcdf/ftp/'
    self.downloadext      = 'tar.gz'
    self.downloadversion  = '4-4.2'
    self.functions        = ['ncCheck']
    self.functionsCxx     = [1, 'namespace netCDF {void ncCheck(int,char*,int);}','netCDF::ncCheck(0,0,0);']
    self.includes         = ['ncCheck.h']
    self.liblist          = [['libnetcdf_c++4.a']]
    self.cxx              = 1
    return

  def setupDownload(self):
    '''Need this because the default puts a '-' between the name and the version number'''
    self.download = [self.downloadpath+self.downloadname+self.downloadversion+'.'+self.downloadext]

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi    = framework.require('config.packages.MPI', self)
    self.hdf5   = framework.require('config.packages.hdf5', self)
    self.netcdf = framework.require('config.packages.netcdf', self)
    self.odeps  = [self.mpi, self.hdf5, self.netcdf]
    self.deps   = [self.mpi]
    return

  def formGNUConfigureExtraArgs(self):
    '''Specify archiver, libdir, disable DAP and HDF4, enable NetCDF4'''
    args = []
    args.append('AR="'+self.setCompilers.AR+'"')
    args.append('ARFLAGS="'+self.setCompilers.AR_FLAGS+'"')
    args.append('--libdir='+os.path.join(self.installDir,self.libdir))
    return args
