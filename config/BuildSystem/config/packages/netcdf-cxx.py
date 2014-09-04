import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download     = ['https://github.com/Unidata/netcdf-cxx4/archive/v4.2.1.tar.gz']
    self.functions    = ['ncCheck']
    self.functionsCxx = [1, 'namespace netCDF {void ncCheck(int,char*,int);}','netCDF::ncCheck(0,0,0);']
    self.includes     = ['ncCheck.h']
    self.liblist      = [['libnetcdf_c++4.a']]
    self.cxx          = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi    = framework.require('config.packages.MPI', self)
    self.hdf5   = framework.require('config.packages.hdf5', self)
    self.netcdf = framework.require('config.packages.netcdf', self)
    self.odeps  = [self.mpi, self.hdf5, self.netcdf]
    self.deps   = [self.mpi]
    return

