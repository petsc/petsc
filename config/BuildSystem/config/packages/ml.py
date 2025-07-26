import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version           = '13.2.0'
    self.gitcommit         = 'v{0}'.format(self.version)
    self.versionname       = 'PACKAGE_VERSION'
    self.download          = ['git://https://gitlab.com/petsc/pkg-trilinos-ml',
                              'https://gitlab.com/petsc/pkg-trilinos-ml/-/archive/{0}/pkg-trilinos-ml-{0}.tar.gz'.format(self.gitcommit)]
    self.functions         = ['ML_Set_PrintLevel']
    self.includes          = ['ml_include.h']
    self.liblist           = [['libml.a']]
    self.license           = 'https://trilinos.github.io'
    self.buildLanguages    = ['Cxx']
    self.precisions        = ['double']
    self.complex           = 0
    self.downloadonWindows = 1
    self.requires32bitint  = 1;  # ml uses a combination of "global" indices that can be 64-bit and local indices that are always int therefore it is
                                 # essentially impossible to use ML's 64-bit integer mode with PETSc's --with-64-bit-indices
    self.hastests          = 1
    self.downloaddirnames  = ['pkg-trilinos-ml']
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.cxxlibs    = framework.require('config.packages.cxxlibs',self)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.mathlib    = framework.require('config.packages.mathlib',self)
    self.metis      = framework.require('config.packages.METIS',self)
    self.deps       = [self.mpi,self.blasLapack,self.cxxlibs,self.mathlib]
    self.odeps      = [self.metis]
    return

  # older versions of Trilinos require passing rpath with the various library paths
  # this caused problems on Apple with CMake generating command lines that are too long
  # Trilinos was fixed to handle the rpath internally using CMake
  def toStringNoDupes(self,string):
    string    = self.libraries.toStringNoDupes(string)
    if self.requiresrpath: return string
    newstring = ''
    for i in string.split(' '):
      if i.find('-rpath') == -1:
        newstring = newstring+' '+i
    return newstring.strip()

  def toString(self,string):
    string    = self.libraries.toString(string)
    if self.requiresrpath: return string
    newstring = ''
    for i in string.split(' '):
      if i.find('-rpath') == -1:
        newstring = newstring+' '+i
    return newstring.strip()

  def formCMakeConfigureArgs(self):
    if '++' in self.externalPackagesDir:
      raise RuntimeError('Cannot build ml in a folder containing "++"')
    self.requiresrpath    = 1
    #  Get trilinos version
    # if version is 120900 (Dev) or higher than don't require rpaths
    trequires = 0
    fd = open(os.path.join(self.packageDir,'Version.cmake'))
    bf = fd.readline()
    while bf:
      if bf.startswith('SET(Trilinos_MAJOR_MINOR_VERSION'):
        bf = bf[34:39]
        bf = int(bf)
        if bf > 120900:
          self.requiresrpath = 0
        if bf == 120900:
          trequires = 1
      if trequires:
        if bf.find('(Dev)') > -1:
          self.requiresrpath = 0
      bf = fd.readline()
    fd.close()
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF')
    args.append('-DTrilinos_ENABLE_ALL_PACKAGES=OFF')
    args.append('-DTrilinos_ENABLE_ML=ON')
    args.append('-DTPL_BLAS_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DTPL_LAPACK_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DBUILD_SHARED_LIBS=ON')
    args.append('-DTPL_ENABLE_MPI=ON')
    if self.metis.found:
      args.append('-DTPL_ENABLE_METIS=ON')
      args.append('-DTPL_METIS_LIBRARIES="'+self.toStringNoDupes(self.metis.lib)+'"')
      args.append('-DTPL_METIS_INCLUDE_DIRS="'+self.headers.toStringNoDupes(self.metis.include)[2:]+'"')
    if not hasattr(self.compilers, 'FC'):
      args.append('-DTrilinos_ENABLE_Fortran=OFF')

    return args
