import config.base
import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version = '4.3.0'
    self.gitcommit = 'v' + self.version
    self.download = ['git://https://github.com/cgns/cgns', 'https://github.com/cgns/cgns/archive/{}.tar.gz'.format(self.gitcommit)]
    self.functions  = ['cgp_close']
    self.includes   = ['cgnslib.h']
    self.liblist    = [['libcgns.a'],
                       ['libcgns.a', 'libhdf5.a']] # Sometimes they over-link

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.hdf5 = framework.require('config.packages.hdf5', self)
    self.mpi = framework.require('config.packages.MPI',self)
    self.deps = [self.hdf5]
    self.odeps = [self.mpi]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DCGNS_BUILD_SHARED:BOOL=ON')
    if not self.mpi.usingMPIUni:
      args.append('-DCGNS_ENABLE_PARALLEL:BOOL=ON')
      args.append('-DHDF5_NEED_MPI:BOOL=ON')
    if self.hdf5.directory:
      args.append('-DHDF5_ROOT:PATH={}'.format(self.hdf5.directory))
    return args
