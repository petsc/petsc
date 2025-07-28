import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version           = '2025.05.28'
    self.gitcommit         = 'v{0}'.format(self.version)
    self.versionname       = 'PACKAGE_VERSION'
    self.download          = ['git://https://github.com/icl-utk-edu/slate']
    self.functionsCxx      = [1,'namespace sl = slate;','sl::gemm()']
    self.includes          = ['slate/slate.hh']
    self.liblist           = [['libslate.a','libblaspp.a','liblapackpp.a']]
    # self.extraLib          = ['mpi_cxx']
    self.license           = 'https://github.com/icl-utk-edu/slate'
    self.buildLanguages    = ['Cxx','FC']
    self.precisions        = ['double']
    self.downloadonWindows = 1
    self.hastests          = 1
    self.gitsubmodules     = ['blaspp','lapackpp']
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.cxxlibs    = framework.require('config.packages.cxxlibs',self)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.scalapack  = framework.require('config.packages.ScaLAPACK',self)
    self.mathlib    = framework.require('config.packages.mathlib',self)
    self.openmp     = framework.require('config.packages.OpenMP',self)
    self.cuda       = framework.require('config.packages.CUDA',self)
    self.hip        = framework.require('config.packages.HIP',self)
    self.deps       = [self.mpi,self.blasLapack,self.cxxlibs,self.mathlib,self.openmp]
    self.odeps      = [self.cuda,self.hip]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)

    if self.cuda.found:
      args.append('-Dgpu_backend=cuda')
      args.extend(self.cuda.getCmakeCUDAArchFlag())
    elif self.hip.found:
      args.append('-Dgpu_backend=hip')
      args.append('-DCMAKE_HIP_ARCHITECTURES="'+self.hip.hipArch+'"') # cmake supports format like "gfx801;gfx900"

    args.append('-DBLAS_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DLAPACK_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')

    args.append('-Dbuild_tests=false')

    return args
