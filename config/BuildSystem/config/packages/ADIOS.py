import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit         = 'master'
    self.download          = ['git://https://github.com/ornladios/ADIOS.git']
    self.downloaddirnames  = ['adios']
    self.functions         = ['adios_open']
    self.includes          = ['adios.h']
    self.liblist           = [['libadiosf.a', 'libadios.a'],['libadios.a']]
    self.hastests          = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.zlib           = framework.require('config.packages.zlib',self)
    self.mpi            = framework.require('config.packages.MPI', self)
    self.hdf5           = framework.require('config.packages.hdf5', self)
    self.netcdf         = framework.require('config.packages.netcdf', self)
    self.deps           = [self.mpi]
    self.odeps          = [self.hdf5,self.netcdf,self.zlib]
    return

  def gitPreReqCheck(self):
    '''ADIOS from the git repository needs the GNU autotools'''
    return self.programs.autoreconf and self.programs.libtoolize

  def formGNUConfigureArgs(self):
    '''Add ADIOS specific configure arguments'''
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--with-mpi="'+self.mpi.directory+'"')
    if self.hdf5.found:
      args.append('--with-phdf5="'+self.hdf5.directory+'"')
    if self.netcdf.found:
      args.append('--with-nc4par="'+self.netcdf.directory+'"')
    if self.zlib.found:
      args.append('--with-zlib="'+self.zlib.directory+'"')
    if not hasattr(self.compilers, 'FC'):
      args.append('--disable-fortran')

    return args

