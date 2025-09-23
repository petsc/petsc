import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.minversion       = '1.8'
    self.versionname      = 'H5_VERSION'
    self.version          = '1.14.6'
    self.download         = ['https://github.com/HDFGroup/hdf5/archive/hdf5_'+self.version+'/hdf5-'+self.version+'.tar.gz',
                             'https://web.cels.anl.gov/projects/petsc/download/externalpackages/hdf5-'+self.version+'.tar.gz']
# David Moulton reports that HDF5 configure can fail on NERSC systems and this can be worked around by removing the
#   getpwuid from the test for ac_func in gethostname getpwuid getrusage lstat
    self.functions        = ['H5T_init']
    self.includes         = ['hdf5.h']
    self.liblist          = [['libhdf5_hl.a', 'libhdf5.a']]
    self.hastests         = 1
    self.precisions       = ['single','double']
    self.installwithbatch = 0

  def setupHelp(self, help):
    config.package.CMakePackage.setupHelp(self, help)
    import nargs
    # PETSc does not need the Fortran/CXX interface.
    # We currently need it to be disabled by default as HDF5 has bugs in their build process as of hdf5-1.12.0.
    # Not all dependencies for Fortran bindings are given in the makefiles, hence a parallel build can fail
    # when it starts a Fortran file before all its needed modules are finished.
    # Barry has reported this to them and they acknowledged it.
    help.addArgument('HDF5', '-with-hdf5-fortran-bindings', nargs.ArgBool(None, 0, 'Use/build HDF5 Fortran interface (PETSc does not need it)'))
    help.addArgument('HDF5', '-with-hdf5-cxx-bindings', nargs.ArgBool(None, 0, 'Use/build HDF5 Cxx interface (PETSc does not need it)'))

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.mathlib        = framework.require('config.packages.mathlib',self)
    self.zlib           = framework.require('config.packages.zlib',self)
    self.szlib          = framework.require('config.packages.szlib',self)
    self.flibs          = framework.require('config.packages.flibs',self)
    self.deps           = [self.mathlib]
    self.odeps          = [self.mpi, self.zlib,self.szlib,self.flibs]
    return

  def applyPatches(self):
    try:
      with open(self.packageDir+'/config/cmake/HDFMacros.cmake') as f_in:
        content = f_in.readlines()
      with open(self.packageDir+'/config/cmake/HDFMacros.cmake','w') as f_out:
        f_out.writelines(c.replace('(CMAKE_DEBUG_POSTFIX "_debug")','(CMAKE_DEBUG_POSTFIX "")') for c in content)
    except:
      self.logPrintWarning("Patching HDF5 failed! Continuing with build")

  def versionToStandardForm(self,ver):
    '''HDF5 indicates patches by appending a -patch<n> after the regular part of the version'''
    return ver.replace('-patch','.')

  def formCMakeConfigureArgs(self):
    ''' Add HDF5 specific --enable-parallel flag and enable Fortran if available '''
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DHDF5_BUILD_HL_LIB=ON')
    args.append('-DHDF5_BUILD_TOOLS=OFF')
    args.append('-DBUILD_TESTING=OFF')

    if not self.mpi.usingMPIUni:
      args.append('-DHDF5_ENABLE_PARALLEL=ON')
    if self.argDB['with-hdf5-fortran-bindings']:
      if hasattr(self.compilers, 'FC'):
        args.append('-DHDF5_BUILD_FORTRAN=ON')
      else:
        raise RuntimeError('Cannot build HDF5 Fortran bindings --with-fc=0 or with a malfunctioning Fortran compiler.')
    if self.argDB['with-hdf5-cxx-bindings']:
      if hasattr(self.compilers, 'CXX'):
        args.extend(['-DHDF5_BUILD_CPP_LIB=ON', '-DALLOW_UNSUPPORTED=ON'])
      else:
        raise RuntimeError('Cannot build HDF5 Cxx bindings --with-cxx=0 or with a malfunctioning Cxx compiler.')

    args.append('-DHDF5_ENABLE_Z_LIB_SUPPORT='+('ON' if self.zlib.found else 'OFF'))
    args.append('-DHDF5_ENABLE_SZIP_SUPPORT='+('ON' if self.szlib.found else 'OFF'))

    return args

  def configureLibrary(self):
    # PETSc does not need the Fortran/CXX interface, but some users will use them
    # and expect our standard linking to be sufficient.  Thus we try to link the Fortran/CXX
    # libraries, but fall back to linking only C.
    if hasattr(self.compilers, 'FC') and self.argDB['with-hdf5-fortran-bindings']:
      self.liblist = [['libhdf5_hl_fortran.a','libhdf5_fortran.a', 'libhdf5_hl_f90cstub.a', 'libhdf5_f90cstub.a'] + libs for libs in self.liblist] \
                   + [['libhdf5_hl_fortran.a','libhdf5_fortran.a'] + libs for libs in self.liblist] \
                   + [['libhdf5hl_fortran.a','libhdf5_fortran.a'] + libs for libs in self.liblist] \
                   + self.liblist
    if hasattr(self.compilers, 'CXX') and self.argDB['with-hdf5-cxx-bindings']:
      self.liblist = [['libhdf5_hl_cpp.a','libhdf5_cpp.a'] + libs for libs in self.liblist] + self.liblist
    config.package.GNUPackage.configureLibrary(self)

    for i in ['ZLIB_H','SZLIB_H','PARALLEL']:
      oldFlags = self.compilers.CPPFLAGS
      self.compilers.CPPFLAGS += ' '+self.headers.toString(self.include)
      try:
        output = self.outputPreprocess('#include "H5pubconf.h"\n#if defined(H5_HAVE_'+i+')\nfoundbeast\n#endif\n')
      except:
        self.log.write('Unable to run preprocessor to obtain '+i+' information\n')
        self.compilers.CPPFLAGS = oldFlags
        return
      self.compilers.CPPFLAGS = oldFlags
      if output.find('foundbeast') > -1:
        if i.endswith('_H'): i = i[0:-2]
        self.addDefine('HDF5_HAVE_'+i, 1)

