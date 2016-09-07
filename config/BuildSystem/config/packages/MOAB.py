import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    # To track MOAB.git, update gitcommit to 'git describe --always' or 'git rev-parse HEAD'
    self.gitcommit         = '712d944e4d11750e2648220d0093e7ea8d0f5a99' # HEAD of MOAB/petsc branch: Sep 3, 2016
    self.download          = ['https://bitbucket.org/fathomteam/moab.git','http://ftp.mcs.anl.gov/pub/fathom/moab-4.9.2.tar.gz']
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
    self.metis     = framework.require('config.packages.metis',self)
    self.parmetis  = framework.require('config.packages.parmetis',self)
    self.ptscotch  = framework.require('config.packages.PTScotch',self)
    self.zoltan    = framework.require('config.packages.Zoltan', self)
    #self.odeps     = [self.mpi, self.hdf5, self.netcdf, self.metis, self.parmetis, self.ptscotch, self.zoltan]
    self.odeps     = [self.mpi, self.hdf5, self.netcdf, self.metis]
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
    if self.metis.found:
      args.append('--with-metis="'+self.metis.directory+'"')
    if self.parmetis.found:
      args.append('--with-parmetis="'+self.parmetis.directory+'"')
    if self.ptscotch.found:
      args.append('--with-scotch="'+self.ptscotch.directory+'"')
    if self.zoltan.found:
      args.append('--with-zoltan="'+self.zoltan.directory+'"')

    return args

