import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.versionname        = 'RAJA_VERSION_MAJOR.RAJA_VERSION_MINOR.RAJA_VERSION_PATCHLEVEL'
    self.download           = ['https://github.com/LLNL/RAJA/releases/download/v2025.03.2/RAJA-v2025.03.2.tar.gz',
                               'https://web.cels.anl.gov/projects/petsc/download/externalpackages/RAJA-v2025.03.2.tar.gz']
    self.downloaddirnames   = ['raja']
    # TODO: BuildSystem checks C++ headers blindly using CXX. However, when Raja  is compiled by CUDAC, for example, using
    # CXX to compile a Raja code raises an error. As a workaround, we set this field to skip checking headers in includes.
    self.doNotCheckIncludes = 1
    self.includes           = ['RAJA/RAJA.hpp']
    self.liblist            = [['libRAJA.a','libcamp.a'],['libRAJA.a']]
    self.buildLanguages     = ['Cxx']
    self.minCxxVersion      = 'c++14'
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
    self.openmp          = framework.require('config.packages.OpenMP',self)
    self.pthread         = framework.require('config.packages.pthread',self)
    self.cuda            = framework.require('config.packages.CUDA',self)
    self.hip             = framework.require('config.packages.HIP',self)
    self.hwloc           = framework.require('config.packages.hwloc',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.odeps           = [self.mpi,self.openmp,self.hwloc,self.cuda,self.hip,self.pthread]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DUSE_XSDK_DEFAULTS=YES')
    if not self.compilerFlags.debugging:
      args.append('-DXSDK_ENABLE_DEBUG=NO')

    if self.openmp.found:
      args.append('-DENABLE_OPENMP:BOOL=ON')
    else:
      args.append('-DENABLE_OPENMP:BOOL=OFF')

    if self.mpi.found and not self.mpi.usingMPIUni:
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
        # Raja CMake adds the -ccbin and -std therefore remove them from provided options
        # to prevent error from double use
        cuda_flags = self.rmArgsStartsWith(self.rmArgsPair(self.getCompilerFlags().split(' '),['-ccbin']),['-std='])
        cuda_flags = self.updatePackageCUDAFlags(cuda_flags)
        args.append('-DCMAKE_CUDA_FLAGS="{}"'.format(cuda_flags))
      args.extend(self.cuda.getCmakeCUDAArchFlag())

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
