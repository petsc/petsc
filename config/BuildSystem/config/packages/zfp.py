import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version           = '1.0.1'
    self.gitcommit         = '{0}'.format(self.version)
    self.versionname       = 'ZFP_VERSION'
    self.download          = ['git://https://github.com/LLNL/zfp']
    self.functions         = ['zfp_field_2d']
    self.includes          = ['zfp/array2.hpp']
    self.liblist           = [['libzfp.a']]
    self.license           = 'https://github.com/LLNL/zfp/'
    self.buildLanguages    = ['Cxx']
    # self.precisions        = ['double']
    self.complex           = 0
    self.downloadonWindows = 1
    self.hastests          = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.cxxlibs    = framework.require('config.packages.cxxlibs',self)
    self.openmp     = framework.require('config.packages.OpenMP',self)
    # self.cuda       = framework.require('config.packages.CUDA',self)
    self.deps       = [self.cxxlibs]
    # self.odeps      = [self.openmp,self.cuda]
    self.odeps      = [self.openmp]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)

    if self.openmp.found:
      args.append('-DZFP_WITH_OPENMP=ON')
    else:
      args.append('-DZFP_WITH_OPENMP=OFF')

    # if self.cuda.found:
    #   args.append('-DZFP_WITH_CUDA=ON')
    #   args.extend(self.cuda.getCmakeCUDAArchFlag())
    # else:
    #   args.append('-DZFP_WITH_CUDA=OFF')

    return args
