import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.version          = '3.1.1'
    self.versionname      = 'STRUMPACK_VERSION_MAJOR.STRUMPACK_VERSION_MINOR.STRUMPACK_VERSION_PATCH'
    self.versioninclude   = 'StrumpackConfig.hpp'
    self.gitcommit        = 'v'+self.version
    self.download         = ['git://https://github.com/pghysels/STRUMPACK','https://github.com/pghysels/STRUMPACK/archive/v'+self.version+'.tar.gz']
    self.functions        = ['STRUMPACK_init']
    self.includes         = ['StrumpackSparseSolver.h']
    self.liblist          = [['libstrumpack.a']]
    self.minCxxVersion    = 'c++11'
    self.buildLanguages   = ['Cxx','FC']
    self.hastests         = 1
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.blasLapack     = framework.require('config.packages.BlasLapack',self)
    self.scalapack      = framework.require('config.packages.scalapack',self)
    self.metis          = framework.require('config.packages.metis',self)
    self.parmetis       = framework.require('config.packages.parmetis',self)
    self.ptscotch       = framework.require('config.packages.PTScotch',self)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.openmp         = framework.require('config.packages.openmp',self)
    self.deps           = [self.mpi,self.blasLapack,self.scalapack,self.metis]
    self.odeps          = [self.parmetis,self.ptscotch,self.openmp]
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)

    args.append('-DTPL_BLAS_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DTPL_LAPACK_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DTPL_SCALAPACK_LIBRARIES="'+self.libraries.toString(self.scalapack.lib)+'"')

    args.append('-DTPL_METIS_LIBRARIES="'+self.libraries.toString(self.metis.lib)+'"')
    args.append('-DTPL_METIS_INCLUDE_DIRS="'+self.headers.toStringNoDupes(self.metis.include)[2:]+'"')

    if self.parmetis.found:
      args.append('-DTPL_ENABLE_PARMETIS=ON')
      args.append('-DTPL_PARMETIS_LIBRARIES="'+self.libraries.toString(self.parmetis.lib)+'"')
      args.append('-DTPL_PARMETIS_INCLUDE_DIRS="'+self.headers.toStringNoDupes(self.parmetis.include)[2:]+'"')
    else:
      args.append('-DTPL_ENABLE_PARMETIS=OFF')

    if self.ptscotch.found:
      args.append('-DTPL_ENABLE_SCOTCH=ON')
      args.append('-DTPL_SCOTCH_LIBRARIES="'+self.libraries.toString(self.ptscotch.lib)+'"')
      args.append('-DTPL_SCOTCH_INCLUDE_DIRS="'+self.headers.toStringNoDupes(self.ptscotch.include)[2:]+'"')
    else:
      args.append('-DTPL_ENABLE_SCOTCH=OFF')

    if self.openmp.found:
      args.append('-DSTRUMPACK_USE_OPENMP=ON')
    else:
      args.append('-DSTRUMPACK_USE_OPENMP=OFF')

    return args

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if self.argDB['with-'+self.package]:
      # Strumpack requires dlapmr() LAPACK routine
      if not self.blasLapack.checkForRoutine('dlapmr'):
        raise RuntimeError('Strumpack requires the LAPACK routine dlapmr(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it\nTry using --download-fblaslapack=1 option \nIf you are using vecLib on OSX, it does not contain this function.')
      self.log.write('Found dlapmr() in Lapack library as needed by Strumpack\n')
    return
