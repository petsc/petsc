import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    # To track MOAB.git, update gitcommit to 'git describe --always' or 'git rev-parse HEAD'
    self.gitcommit         = '70d39f94e9854a78ea7daeddb6f25832c17c7783' # HEAD of MOAB/petsc branch: Apr 24, 2016
    self.download          = ['https://bitbucket.org/fathomteam/moab.git','http://ftp.mcs.anl.gov/pub/fathom/moab-70d39f94e9854a78ea7daeddb6f25832c17c7783.tar.gz']
    self.downloaddirnames  = ['moab']
    # Check for moab::Core and includes/libraries to verify build
    self.functions         = ['Core']
    self.functionsCxx      = [1, 'namespace moab {class Core {public: Core();};}','moab::Core *mb = new moab::Core()']
    self.includes          = ['moab/Core.hpp']
    self.liblist           = [['libiMesh.a', 'libMOAB.a'],['libMOAB.a']]
    self.cxx               = 1
    self.precisions        = ['single','double']
    self.hastests          = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi       = framework.require('config.packages.MPI', self)
    self.hdf5      = framework.require('config.packages.hdf5', self)
    self.netcdf    = framework.require('config.packages.netcdf', self)
    self.odeps     = [self.mpi, self.hdf5, self.netcdf]
    return

  def gitPreReqCheck(self):
    '''MOAB from the git repository needs the GNU autotools'''
    return self.programs.autoreconf and self.programs.libtoolize

  def formGNUConfigureArgs(self):
    '''Add MOAB specific configure arguments'''
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--with-mpi="'+self.mpi.directory+'"')
    args.append('--enable-tools')
    if self.hdf5.found:
      args.append('--with-hdf5="'+self.hdf5.directory+'"')
    else:
      args.append('--without-hdf5')
    if self.netcdf.found:
      args.append('--with-netcdf="'+self.netcdf.directory+'"')
    else:
      args.append('--without-netcdf')
    return args



