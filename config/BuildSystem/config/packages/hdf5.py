import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    #hdf5-1.10.0-patch1 breaks with default MPI on ubuntu 12.04 [and freebsd/opensolaris]. So use hdf5-1.8.18 for now.
    self.download  = ['https://support.hdfgroup.org/ftp/HDF5/current18/src/hdf5-1.8.18.tar.gz',
                      'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/hdf5-1.8.18.tar.gz']
# David Moulton reports that HDF5 configure can fail on NERSC systems and this can be worked around by removing the
#   getpwuid from the test for ac_func in gethostname getpwuid getrusage lstat
    self.functions = ['H5T_init']
    self.includes  = ['hdf5.h']
    self.liblist   = [['libhdf5_hl.a', 'libhdf5.a']]
    self.complex          = 1
    self.hastests         = 1
    self.precisions       = ['single','double'];
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.mathlib        = framework.require('config.packages.mathlib',self)
    self.zlib           = framework.require('config.packages.zlib',self)
    self.szlib          = framework.require('config.packages.szlib',self)
    self.deps           = [self.mpi,self.mathlib]
    self.odeps          = [self.zlib,self.szlib]
    return

  def formGNUConfigureArgs(self):
    ''' Add HDF5 specific --enable-parallel flag and enable Fortran if available '''
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--with-default-api-version=v18') # for hdf-1.10
    args.append('--enable-parallel')
    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      args.append('--enable-fortran')
      args.append('F9X="'+self.setCompilers.getCompiler()+'"')
      self.setCompilers.popLanguage()
    if self.zlib.found:
      args.append('--with-zlib=yes')
    else:
      args.append('--with-zlib=no')
    if self.szlib.found:
      args.append('--with-szlib=yes')
    else:
      args.append('--with-szlib=no')

    args.append('CPPFLAGS="'+self.headers.toStringNoDupes(self.dinclude)+'"')
    args.append('LIBS="'+self.libraries.toStringNoDupes(self.dlib)+'"')

    return args

  def configureLibrary(self):
    if hasattr(self.compilers, 'FC'):
      # PETSc does not need the Fortran interface, but some users will call the Fortran interface
      # and expect our standard linking to be sufficient.  Thus we try to link the Fortran
      # libraries, but fall back to linking only C.
      self.liblist = [['libhdf5hl_fortran.a','libhdf5_fortran.a'] + libs for libs in self.liblist] + self.liblist
    config.package.GNUPackage.configureLibrary(self)
    if self.libraries.check(self.dlib, 'H5Pset_fapl_mpio'):
      self.addDefine('HAVE_H5PSET_FAPL_MPIO', 1)
    return
