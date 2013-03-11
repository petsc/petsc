import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.downloadpath      = 'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/'
    self.downloadname      = 'moab'
    self.downloadfilename  = 'moab'
    self.downloadversion   = '3.99-10-18-2010'
    self.downloadext       = 'tar.gz'

    self.functions         = ['iMesh_newMesh']
    self.functionsFortran  = 1
    self.includes          = ['iMesh.h']
    self.liblist           = [['libiMesh.a', 'libMOAB.a']]
    self.cxx               = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi             = framework.require('config.packages.MPI', self)
    self.hdf5            = framework.require('config.packages.hdf5', self)
    self.netcdf          = framework.require('config.packages.netcdf', self)
    self.odeps           = [self.mpi, self.hdf5, self.netcdf]
    return


