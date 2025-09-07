import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version          = '2025.09.0'
    self.gitcommit        = 'v'+self.version
    self.download         = ['git://https://github.com/LLNL/Umpire.git',
                             'https://github.com/LLNL/Umpire/archive/'+self.gitcommit+'.tar.gz']
    self.gitsubmodules    = ['blt', 'src/tpl/umpire/camp', 'src/tpl/umpire/fmt']
    self.includes         = ['umpire/Allocator.hpp']
    self.doNotCheckIncludes = 1
    self.liblist          = [['libumpire.a','libcamp.a'], ['libumpire.a']]
    self.buildLanguages   = ['Cxx']
    self.minCxxVersion    = 'c++17'
    self.precisions       = ['single','double']
    self.minCmakeVersion  = (3,23,0)
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.cuda   = framework.require('config.packages.CUDA', self)
    self.hip    = framework.require('config.packages.HIP', self)
    self.sycl   = framework.require('config.packages.SYCL', self)
    self.openmp = framework.require('config.packages.OpenMP', self)
    self.mpi    = framework.require('config.packages.MPI', self)
    self.odeps  = [self.cuda, self.hip, self.sycl, self.openmp, self.mpi]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)

    # Turn on C interface
    args.append('-DUMPIRE_ENABLE_C=ON')

    # Turn off extras to speed the build in PETSc context
    args.append('-DUMPIRE_ENABLE_TOOLS=OFF')
    args.append('-DENABLE_TESTS=OFF')
    args.append('-DENABLE_BENCHMARKS=OFF')
    args.append('-DENABLE_EXAMPLES=OFF')

    # Backends
    args.append('-DENABLE_CUDA={}'.format('ON' if self.cuda.found else 'OFF'))
    args.append('-DENABLE_HIP={}'.format('ON' if self.hip.found else 'OFF'))
    args.append('-DENABLE_SYCL={}'.format('ON' if self.sycl.found else 'OFF'))
    args.append('-DENABLE_OPENMP={}'.format('ON' if self.openmp.found else 'OFF'))
    if hasattr(self, 'mpi') and self.mpi.found and not self.mpi.usingMPIUni:
      args.append('-DENABLE_MPI=ON')
    else:
      args.append('-DENABLE_MPI=OFF')

    # CUDA specifics
    if self.cuda.found:
      # Prefer cmake CUDA arch propagation
      args.extend(self.cuda.getCmakeCUDAArchFlag())

    # HIP specifics
    if self.hip.found:
      # Umpire relies on HIP toolchain; set HIP_ROOT_DIR if available
      if hasattr(self.hip,'hipDir'):
        args.append('-DHIP_ROOT_DIR='+self.hip.hipDir)
      # Set offload arch if available
      if hasattr(self.hip,'hipArch'):
        args.append('-DCMAKE_HIP_ARCHITECTURES='+self.hip.hipArch)
      # Help BLT/CMake locate hip-config.cmake explicitly
      try:
        hip_cmake_candidates = [
          os.path.join(self.hip.hipDir,'lib','cmake','hip'),
          os.path.join(self.hip.hipDir,'hip','lib','cmake','hip'),
        ]
        for cm_dir in hip_cmake_candidates:
          if os.path.isdir(cm_dir):
            args.append('-Dhip_DIR='+cm_dir)
            break
      except Exception:
        pass

    # TODO: SYCL specifics

    return args
