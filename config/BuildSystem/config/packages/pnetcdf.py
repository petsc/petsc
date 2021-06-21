import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version          = '1.12.2'
    self.versionname      = 'PNETCDF_VERSION'
    self.gitcommit        = 'checkpoint.' + self.version # 1.12.1 is first to include MPI1 deprecated fix
    self.download         = ['git://https://github.com/parallel-netcdf/pnetcdf',
                             'https://parallel-netcdf.github.io/Release/pnetcdf-'+self.version+'.tar.gz',
                             'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/pnetcdf-'+self.version+'.tar.gz']
    self.functions        = ['ncmpi_create']
    self.includes         = ['pnetcdf.h']
    self.liblist          = [['libpnetcdf.a']]
    self.useddirectly     = 0
    self.installwithbatch = 0
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.flibs = framework.require('config.packages.flibs',self)
    self.mpi   = framework.require('config.packages.MPI', self)
    self.deps  = [self.mpi,self.flibs]
    return

  def formGNUConfigureArgs(self):
    # https://github.com/Parallel-NetCDF/PnetCDF/commit/38d210c006cabff70d78204d2db98a22ab87547c
    if hasattr(self.mpi,'ompi_version') and self.mpi.ompi_version >= (4,0,0):
        self.minversion = '1.12.1'
        oldinclude = self.include
        self.include.append(os.path.join(self.packageDir,'src','include'))
        self.checkVersion()
        self.include = oldinclude

    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    self.addToArgs(args,'LIBS',self.libraries.toStringNoDupes(self.flibs.lib))
    return args
