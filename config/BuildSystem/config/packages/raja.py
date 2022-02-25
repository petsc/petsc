import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit          = 'v0.14.0'
    self.versionname        = 'RAJA_VERSION_MAJOR.RAJA_VERSION_MINOR.RAJA_VERSION_PATCHLEVEL'
    self.download           = ['git://https://github.com/LLNL/RAJA.git']
    self.gitsubmodules      = ['.']
    self.downloaddirnames   = ['raja']
    # TODO: BuildSystem checks C++ headers blindly using CXX. However, when Raja  is compiled by CUDAC, for example, using
    # CXX to compile a Raja code raises an error. As a workaround, we set this field to skip checking headers in includes.
    self.doNotCheckIncludes = 1
    self.includes           = ['RAJA/RAJA.hpp']
    self.liblist            = [['libRAJA.a']]
    self.buildLanguages     = ['Cxx']
    self.minCxxVersion      = 'c++14'
    self.downloadonWindows  = 0
    self.hastests           = 1
    self.requiresrpath      = 1
    self.precisions         = ['single','double']
    return

  def __str__(self):
    output  = config.package.CMakePackage.__str__(self)
    if hasattr(self,'system'): output += '  Backend: '+self.system+'\n'
    return output

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.externalpackagesdir = framework.require('PETSc.options.externalpackagesdir',self)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.flibs           = framework.require('config.packages.flibs',self)
    self.cxxlibs         = framework.require('config.packages.cxxlibs',self)
    self.mathlib         = framework.require('config.packages.mathlib',self)
    self.deps            = [self.blasLapack,self.flibs,self.cxxlibs,self.mathlib]
    self.openmp          = framework.require('config.packages.openmp',self)
    self.pthread         = framework.require('config.packages.pthread',self)
    self.cuda            = framework.require('config.packages.cuda',self)
    self.hip             = framework.require('config.packages.hip',self)
    self.hwloc           = framework.require('config.packages.hwloc',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.odeps           = [self.mpi,self.openmp,self.hwloc,self.cuda,self.hip,self.pthread]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DUSE_XSDK_DEFAULTS=YES')
    if not self.compilerFlags.debugging:
      args.append('-DXSDK_ENABLE_DEBUG=NO')

    if self.checkSharedLibrariesEnabled():
      args.append('-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON')
      args.append('-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=ON')

    if self.openmp.found:
      args.append('-DENABLE_OPENMP:BOOL=ON')
    else:
      args.append('-DENABLE_OPENMP:BOOL=OFF')

    if self.mpi.found:
      args.append('-DENABLE_MPI=ON')

    # Raja documents these flags though they may not exist
    args.append('-DRAJA_ENABLE_TESTS=OFF')
    args.append('-DRAJA_ENABLE_BENCHMARKS=OFF')
    args.append('-DRAJA_ENABLE_EXAMPLES=OFF')
    args.append('-DRAJA_ENABLE_EXERCISES=OFF')
    # submodule BLT has the following flags
    args.append('-DENABLE_TESTS=OFF')
    args.append('-DENABLE_BENCHMARKS=OFF')
    args.append('-DENABLE_EXAMPLES=OFF')
    args.append('-DENABLE_EXERCISES=OFF')
    args.append('-DENABLE_CTEST=OFF')

    if self.cuda.found:
      args.append('-DENABLE_CUDA=ON')
      self.system = 'CUDA'

      with self.Language('CUDA'):
        args.append('-DCMAKE_CUDA_COMPILER='+self.getCompiler())
        # Raja cmake adds the -ccbin and -std therefor remove them from provided options to prevent error from double use
        args.append('-DCMAKE_CUDA_FLAGS="'+' '.join(self.rmArgsStartsWith(self.rmArgsPair(self.getCompilerFlags().split(' '),['-ccbin']),['-std=']))+'"')

      if hasattr(self.cuda,'cudaArch'):
        generation = 'sm_'+self.cuda.cudaArch
      else:
        raise RuntimeError('You must set --with-cuda-arch=60, 70, 75, 80 etc.')
      args.append('-DCUDA_ARCH='+generation)

    elif self.hip.found:
      raise RuntimeError('No support in downloader for HIP')

    return args

  def configureLibrary(self):
    import os
    config.package.CMakePackage.configureLibrary(self)
    if self.cuda.found:
      self.addMakeMacro('RAJA_USE_CUDA_COMPILER',1)
    elif self.hip.found:
      self.addMakeMacro('RAJA_USE_HIP_COMPILER',1)

