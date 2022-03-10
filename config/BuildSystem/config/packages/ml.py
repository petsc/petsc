import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version           = '13.2.0'
    self.gitcommit         = 'v{0}'.format(self.version)
    self.versionname       = 'PACKAGE_VERSION'
    self.download          = ['git://https://gitlab.com/petsc/pkg-trilinos-ml',
                              'https://gitlab.com/petsc/pkg-trilinos-ml/-/archive/{0}/pkg-trilinos-ml-{0}.tar.gz'.format(self.gitcommit)]
    self.functions         = ['ML_Set_PrintLevel']
    self.includes          = ['ml_include.h']
    self.liblist           = [['libml.a']]
    self.license           = 'https://trilinos.github.io'
    self.buildLanguages    = ['Cxx']
    self.precisions        = ['double']
    self.complex           = 0
    self.downloadonWindows = 1
    self.requires32bitint  = 1;  # ml uses a combination of "global" indices that can be 64 bit and local indices that are always int therefore it is
                                 # essentially impossible to use ML's 64 bit integer mode with PETSc's --with-64-bit-indices
    self.hastests          = 1
    self.downloaddirnames  = ['pkg-trilinos-ml']
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.cxxlibs    = framework.require('config.packages.cxxlibs',self)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.mathlib    = framework.require('config.packages.mathlib',self)
    self.deps       = [self.mpi,self.blasLapack,self.cxxlibs,self.mathlib]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF')
    args.append('-DTrilinos_ENABLE_ALL_PACKAGES=OFF')
    args.append('-DTrilinos_ENABLE_ML=ON')
    args.append('-DTPL_BLAS_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DTPL_LAPACK_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DBUILD_SHARED_LIBS=ON')
    args.append('-DTPL_ENABLE_MPI=ON')
    if not hasattr(self.compilers, 'FC'):
      args.append('-DTrilinos_ENABLE_Fortran=OFF')

    return args
