import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    #hdf5-1.10.0-patch1 breaks with default MPI on ubuntu 12.04 [and freebsd/opensolaris]. So use hdf5-1.8.18 for now.
    self.download  = ['https://support.hdfgroup.org/ftp/HDF5/prev-releases/hdf5-1.8/hdf5-1.8.18/src/hdf5-1.8.18.tar.gz',
                      'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/hdf5-1.8.18.tar.gz']
# David Moulton reports that HDF5 configure can fail on NERSC systems and this can be worked around by removing the
#   getpwuid from the test for ac_func in gethostname getpwuid getrusage lstat
    self.functions = ['H5T_init']
    self.includes  = ['hdf5.h']
    self.liblist   = [['libhdf5_hl.a', 'libhdf5.a']]
    self.complex          = 1
    self.hastests         = 1
    self.precisions       = ['single','double'];
    self.hdf5_major_version    = ''
    self.hdf5_minor_version    = ''
    self.hdf5_release_version  = ''
    self.hdf5_version          = ''
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

  def checkVersion(self):
    import re
    HASHLINESPACE = ' *(?:\n#.*\n *)*'
    oldFlags = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
    hdf5_test = '#include <hdf5.h>\nint hdf5_major = H5_VERS_MAJOR;\nint hdf5_minor = H5_VERS_MINOR;\nint hdf5_release = H5_VERS_RELEASE;\n'
    if self.checkCompile(hdf5_test):
      buf = self.outputPreprocess(hdf5_test)
      try:
        self.hdf5_major_version = re.compile('\nint hdf5_major ='+HASHLINESPACE+'([0-9]+)'+HASHLINESPACE+';').search(buf).group(1)
        self.hdf5_minor_version = re.compile('\nint hdf5_minor ='+HASHLINESPACE+'([0-9]+)'+HASHLINESPACE+';').search(buf).group(1)
        self.hdf5_release_version = re.compile('\nint hdf5_release ='+HASHLINESPACE+'([0-9]+)'+HASHLINESPACE+';').search(buf).group(1)
        self.hdf5_version = str(self.hdf5_major_version) + '.' + str(self.hdf5_minor_version) + '.' + str(self.hdf5_release_version)
      except:
        self.logPrint('Unable to parse HDF5 version from header. Probably a buggy preprocessor')
      if self.hdf5_minor_version and int(self.hdf5_minor_version) < 8:
        raise RuntimeError('HDF5 version must be at least 1.8.0; yours is ' + self.hdf5_version)
      self.addDefine('HAVE_HDF5_MAJOR_VERSION',self.hdf5_major_version)
      self.addDefine('HAVE_HDF5_MINOR_VERSION',self.hdf5_minor_version)
      self.addDefine('HAVE_HDF5_RELEASE_VERSION',self.hdf5_release_version)
    self.compilers.CPPFLAGS = oldFlags
    return

  def configureLibrary(self):
    if hasattr(self.compilers, 'FC'):
      # PETSc does not need the Fortran interface, but some users will call the Fortran interface
      # and expect our standard linking to be sufficient.  Thus we try to link the Fortran
      # libraries, but fall back to linking only C.
      self.liblist = [['libhdf5hl_fortran.a','libhdf5_fortran.a'] + libs for libs in self.liblist] + self.liblist
    config.package.GNUPackage.configureLibrary(self)
    self.checkVersion()
    if self.libraries.check(self.dlib, 'H5Pset_fapl_mpio'):
      self.addDefine('HAVE_H5PSET_FAPL_MPIO', 1)
    return
