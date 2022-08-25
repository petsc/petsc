import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version          = ''
    self.versionname      = ''
    self.gitcommit        = '888c206e7596fe926ea05d7121de4cbb4e9ff90f' # main Aug 22 2022
    self.download         = ['git://https://github.com/NVIDIA/AMGX']
    self.functions        = []
    self.includes         = ['amgx_c.h']
    self.liblist          = [['libamgx.a']]
    self.precisions       = ['double']
    self.cxx              = 1
    self.requires32bitint = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.cuda           = framework.require('config.packages.cuda',self)
    self.deps           = [self.mpi,self.cuda]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    with self.Language('CUDA'):
      args.append('-DCMAKE_CUDA_COMPILER="'+self.getCompiler()+'"')
      if self.compilerFlags.debugging:
        args.append('-DCMAKE_BUILD_TYPE=RelWithTraces')
      else:
        args.append('-DCMAKE_BUILD_TYPE=Release')
      #args.append('-DCMAKE_CXX_FLAGS="-O3"')
      #args.append('-DCMAKE_C_FLAGS="-O3"')
      args.append('-DCUDA_ARCH=' + self.cuda.cudaArch)
      args.append('-DCUDAToolkit_ROOT=' + self.cuda.cudaDir)
    with self.Language('C++'):
      args.append('\'-DCMAKE_CUDA_HOST_COMPILER='+self.framework.getCompiler('Cxx')+'\'')
    return args
