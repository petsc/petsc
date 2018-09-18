import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit         = 'master'
    self.download          = ['git://https://github.com/ornladios/ADIOS2.git']
    self.downloaddirnames  = ['adios2']
    self.functions         = ['adios2_open']
    self.includes          = ['adios2_c.h']
    self.liblist           = [['libadios2f.a', 'libadios2.a'],['libadios2.a']]
    self.hastests          = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.mpi            = framework.require('config.packages.MPI', self)
    self.hdf5           = framework.require('config.packages.hdf5', self)
    self.deps           = [self.mpi]
    self.odeps          = [self.hdf5]
    return

  def formCMakeConfigureArgs(self):
    '''Add ADIOS2 specific configure arguments'''
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DADIOS2_USE_MPI=ON')
    if self.hdf5.found:
      args.append('-DADIOS2_USE_HDF5=ON')
      args.append('-DHDF5_DIR="'+self.hdf5.directory+'"')
    else:
      args.append('-DADIOS2_USE_HDF5=OFF')
    if not hasattr(self.compilers, 'FC'):
      args.append('-DADIOS2_USE_Fortran=OFF')

    return args

