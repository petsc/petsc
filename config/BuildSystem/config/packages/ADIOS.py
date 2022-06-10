import config.package

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit         = 'de85222e71148d2ae47e6af3f9e48dc98323cffb' # master Aug 4, 2021
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
    self.pthread        = framework.require('config.packages.pthread', self)
    self.mathlib        = framework.require('config.packages.mathlib',self)
    self.deps           = [self.mpi,self.pthread,self.mathlib]
    self.odeps          = [self.hdf5,self.netcdf,self.zlib]
    return

  def gitPreReqCheck(self):
    '''ADIOS from the git repository needs the GNU autotools'''
    return self.programs.autoreconf and self.programs.libtoolize

  def formGNUConfigureArgs(self):
    '''Add ADIOS specific configure arguments'''
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    self.pushLanguage('C')
    args.append('MPICC="'+self.getCompiler()+'"')
    self.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.pushLanguage('Cxx')
      args.append('MPICXX="'+self.getCompiler()+'"')
      self.popLanguage()
    if hasattr(self.compilers, 'FC'):
      self.pushLanguage('FC')
      args.append('MPIFC="'+self.getCompiler()+'"')
      self.popLanguage()
    else:
      args.append('--disable-fortran')
    if self.hdf5.found:
      args.append('--with-phdf5=yes')
      args.append('--with-phdf5-incdir='+self.installDir)
      args.append('--with-phdf5-libdir='+self.installDir)
      args.append('--with-phdf5-libs=" "')
    if self.netcdf.found:
      args.append('--with-nc4par=yes')
      args.append('--with-nc4par-incdir='+self.installDir)
      args.append('--with-nc4par-libdir='+self.installDir)
      args.append('--with-nc4par-libs=" "')
      args.append('NETCDF_LIBS=" "')
    if self.zlib.found:
      args.append('--with-zlib=yes')
      args.append('--with-zlib-incdir='+self.installDir)
      args.append('--with-zlib-libdir='+self.installDir)
      args.append('--with-zlib-libs=" "')
      args.append('ZLIB_LIBS=" "')

    args.append('CPPFLAGS="'+self.headers.toStringNoDupes(self.dinclude)+'"')
    args.append('LIBS="'+self.libraries.toStringNoDupes(self.dlib)+'"')

    return args

