import config.package
import os

class Configure(config.package.PythonPackage):
  def __init__(self, framework):
    config.package.PythonPackage.__init__(self, framework)
    self.pkgname                = 'firedrake'
    self.useddirectly           = 0
    self.linkedbypetsc          = 0
    self.builtafterpetsc        = 1
    self.PrefixWriteCheck       = 0

  def setupDependencies(self, framework):
    config.package.PythonPackage.setupDependencies(self, framework)
    self.python          = framework.require('config.packages.Python',self)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.installdir      = framework.require('PETSc.options.installDir',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.slepc           = framework.require('config.packages.SLEPc',self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.petsc4py        = framework.require('config.packages.petsc4py',self)
    self.fftw            = framework.require('config.packages.fftw',self)
    self.hwloc           = framework.require('config.packages.hwloc',self)
    self.hdf5            = framework.require('config.packages.HDF5',self)
    self.metis           = framework.require('config.packages.METIS',self)
    self.pnetcdf         = framework.require('config.packages.PnetCDF',self)
    self.scalapack       = framework.require('config.packages.ScaLAPACK',self)
    self.suitesparse     = framework.require('config.packages.SuiteSparse',self)
    self.zlib            = framework.require('config.packages.zlib',self)
    self.bison           = framework.require('config.packages.Bison',self)
    self.pybind11        = framework.require('config.packages.pybind11',self)
    self.ptscotch        = framework.require('config.packages.PTSCOTCH',self)
    self.mumps           = framework.require('config.packages.MUMPS',self)
    self.netcdf          = framework.require('config.packages.netCDF',self)
    self.superlu_dist    = framework.require('config.packages.SuperLU_DIST',self)
    self.hypre           = framework.require('config.packages.hypre',self)
    self.rtree           = framework.require('config.packages.rtree',self)
    self.libsupermesh    = framework.require('config.packages.libsupermesh',self)
    self.deps            = [self.mpi,self.blasLapack,self.petsc4py,self.slepc,self.fftw,self.hwloc,self.hdf5,self.metis,self.pnetcdf,self.scalapack,self.suitesparse,self.zlib,self.bison,self.ptscotch,self.mumps,self.netcdf,self.superlu_dist,self.hypre,self.pybind11,self.rtree,self.libsupermesh]

  def Install(self):
    self.env = {}
    self.env['PETSC_DIR']  = self.petscdir.dir
    self.env['PETSC_ARCH'] = self.arch
    self.env['HDF5_MPI']   = 'ON'
    self.env['HDF5_DIR']   = self.hdf5.directory
    self.env['SLEPC_DIR']  = self.slepc.directory
    self.pkgbuild          = "firedrake[slepc]"
    return config.package.PythonPackage.Install(self)
