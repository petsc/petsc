import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version          = '3.2.0'
    self.versionname      = 'PACKAGE_VERSION'
    self.gitcommit        = 'v'+self.version
    self.download         = ['git://https://github.com/liuyangzhuan/ButterflyPACK']
    self.functions        = ['s_c_bpack_construct_init']
    self.includes         = ['sC_BPACK_wrapper.h']
    self.liblist          = [['libsbutterflypack.a','libdbutterflypack.a','libcbutterflypack.so','libzbutterflypack.a']]
    self.buildLanguages   = ['FC','Cxx']
    self.hastests         = 1
    self.minCmakeVersion  = (3,3,0)
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.blasLapack     = framework.require('config.packages.BlasLapack',self)
    self.scalapack      = framework.require('config.packages.ScaLAPACK',self)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.openmp         = framework.require('config.packages.OpenMP',self)
    self.deps           = [self.mpi,self.blasLapack,self.scalapack]
    self.odeps          = [self.openmp]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)

    args.append('-DTPL_BLAS_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DTPL_LAPACK_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DTPL_SCALAPACK_LIBRARIES="'+self.libraries.toString(self.scalapack.lib)+'"')

    if self.openmp.found:
      args.append('-Denable_openmp=ON')
    else:
      args.append('-Denable_openmp=OFF')

    return args
