import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version          = ''
    self.versionname      = ''
    self.download         = ['https://web.cels.anl.gov/projects/petsc/download/externalpackages/amgx-2.4.0.tar.gz']
    self.gitsubmodules    = ['.']
    self.functions        = []
    self.includes         = ['amgx_c.h']
    self.liblist          = [['libamgx.a']]
    self.precisions       = ['double']
    self.cxx              = 1
    self.requires32bitint = 1
    self.maxCxxVersion    = 'c++17' # https://github.com/NVIDIA/AMGX/issues/231
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.cuda           = framework.require('config.packages.CUDA',self)
    self.deps           = [self.mpi,self.cuda]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    if self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=RelWithTraces')
    #args.append('-DCMAKE_CXX_FLAGS="-O3"')
    #args.append('-DCMAKE_C_FLAGS="-O3"')
    args.extend(self.cuda.getCmakeCUDAArchFlag())
    if not hasattr(self.cuda, 'cudaDir'):
      raise RuntimeError('CUDA directory not detected! Mail configure.log to petsc-maint@mcs.anl.gov.')
    args.append('-DCUDAToolkit_ROOT=' + self.cuda.cudaDir)
    return args
