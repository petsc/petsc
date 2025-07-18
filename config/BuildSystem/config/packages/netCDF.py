import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version          = '4.9.2'
    self.versionname      = 'NC_VERSION_MAJOR.NC_VERSION_MINOR.NC_VERSION_PATCH'
    self.versioninclude   = 'netcdf_meta.h'
    self.download         = ['https://web.cels.anl.gov/projects/petsc/download/externalpackages/netcdf-c-4.9.2-p1.tar.gz']
    self.functions        = ['nccreate']
    self.includes         = ['netcdf.h']
    self.liblist          = [['libnetcdf.a']]
    self.useddirectly     = 0
    self.installwithbatch = 0
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi     = framework.require('config.packages.MPI', self)
    self.pnetcdf = framework.require('config.packages.PnetCDF', self)
    self.hdf5    = framework.require('config.packages.HDF5', self)
    self.zlib    = framework.require('config.packages.zlib',self)
    self.deps    = [self.mpi, self.hdf5,self.zlib]
    self.odeps   = [self.pnetcdf]
    return

  def formGNUConfigureArgs(self):
    ''' disable DAP and HDF4, enable NetCDF4'''
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('CPPFLAGS="'+self.headers.toString(self.dinclude)+'"')
    self.addToArgs(args,'LIBS',self.libraries.toString(self.dlib)+' '+self.compilers.LIBS)
    args.append('--enable-netcdf-4')
    if self.pnetcdf.found:
      args.append('--enable-pnetcdf')
    args.append('--disable-dap')
    args.append('--disable-dynamic-loading') #This was disabled in v4.3.2 - but enabled in subsequent versions - giving config errors on freebsd (wrt -ldl)
    args.append('--disable-libxml2')
    args.append('--disable-byterange') # curl/curl.h required for byte range support
    args.append('--disable-hdf4')
    return args
