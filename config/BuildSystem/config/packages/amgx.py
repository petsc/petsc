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
    self.maxCxxVersion    = 'c++17' # https://github.com/NVIDIA/AMGX/issues/231
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.cuda           = framework.require('config.packages.cuda',self)
    self.deps           = [self.mpi,self.cuda]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    if self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=RelWithTraces')
    #args.append('-DCMAKE_CXX_FLAGS="-O3"')
    #args.append('-DCMAKE_C_FLAGS="-O3"')
    args.append('-D'+self.cuda.cmakeArchProperty())
    args.append('-DCUDAToolkit_ROOT=' + self.cuda.cudaDir)
    return args
