import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.downloadname    = 'netcdf'
    self.downloadpath    = 'http://www.unidata.ucar.edu/downloads/netcdf/ftp/'
    self.downloadext     = 'tar.gz'
    self.downloadversion = '4.0.1'
    self.functions = ['nccreate']
    self.includes  = ['netcdf.h']
    self.liblist   = [['libnetcdf_c++.a','libnetcdf.a']]
    self.cxx       = 1
    return



  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi             = framework.require('config.packages.MPI', self)
    self.hdf5            = framework.require('PETSc.packages.hdf5', self)
    self.odeps           = [self.mpi, self.hdf5]
    return



