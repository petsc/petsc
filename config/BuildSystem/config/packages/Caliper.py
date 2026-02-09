import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    '''Configure the Caliper performance profiling library, which can be used by the hypre package.'''
    config.package.CMakePackage.__init__(self, framework)
    self.version          = '2.14.0'
    self.versionname      = 'CALIPER_VERSION'
    self.versioninclude   = ['caliper/caliper-config.h']
    self.gitcommit        = 'v'+self.version
    self.download         = ['git://https://github.com/LLNL/Caliper.git',
                             'https://github.com/LLNL/Caliper/archive/v'+self.version+'.tar.gz']
    self.includes         = ['caliper/cali.h']
    self.liblist          = [['libcaliper.a'],['caliper.lib']]
    self.buildLanguages   = ['Cxx']
    self.minCmakeVersion  = (3,12,0)
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.mpi   = framework.require('config.packages.MPI',self)
    self.cuda  = framework.require('config.packages.CUDA', self)
    self.odeps = [self.mpi, self.cuda]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    if self.mpi.found and not self.mpi.usingMPIUni:
      args.append('-DWITH_MPI=ON')

    if self.cuda.found:
      args.extend(self.cuda.getCmakeCUDAArchFlag())
      args.append('-DWITH_CUPTI=ON')
      args.append('-DWITH_NVTX=ON')

    return args
