import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit = 'v2.11.0-1-g5aeeaec'
    self.download  = ['git://https://github.com/LLNL/hypre','https://github.com/LLNL/hypre/archive/v2.11.0.tar.gz']
    self.functions = ['HYPRE_IJMatrixCreate']
    self.includes  = ['HYPRE.h']
    self.liblist   = [['libHYPRE.a']]
    self.license   = 'https://computation.llnl.gov/casc/linear_solvers/sls_hypre.html'
    # Per hypre users guide section 7.5 - install manually on windows for MS compilers.
    self.downloadonWindows = 0
    self.double            = 1
    self.complex           = 0
    self.hastests          = 1
    self.hastestsdatafiles = 1

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.openmp         = framework.require('config.packages.openmp',self)
    self.indexTypes     = framework.require('PETSc.options.indexTypes', self)
    self.languages      = framework.require('PETSc.options.languages',   self)
    self.blasLapack     = framework.require('config.packages.BlasLapack',self)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.deps           = [self.mpi,self.blasLapack]

  def generateLibList(self,dir):
    '''Normally the one in package.py is used, but hypre requires the extra C++ library'''
    alllibs = config.package.GNUPackage.generateLibList(self,dir)
    if self.languages.clanguage == 'C':
      alllibs[0].extend(self.compilers.cxxlibs)
    return alllibs

  def formGNUConfigureArgs(self):
    self.packageDir = os.path.join(self.packageDir,'src')
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    if not hasattr(self.compilers, 'CXX'):
      raise RuntimeError('Error: Hypre requires C++ compiler. None specified')
    if not hasattr(self.compilers, 'FC'):
      args.append('--disable-fortran')
    if self.mpi.include:
      # just use the first dir - and assume the subsequent one isn't necessary [relavant only on AIX?]
      args.append('--with-MPI-include="'+self.mpi.include[0]+'"')
    libdirs = []
    for l in self.mpi.lib:
      ll = os.path.dirname(l)
      libdirs.append(ll)
    libdirs = ' '.join(libdirs)
    args.append('--with-MPI-lib-dirs="'+libdirs+'"')
    libs = []
    for l in self.mpi.lib:
      ll = os.path.basename(l)
      libs.append(ll[3:-2])
    libs = ' '.join(libs)
    args.append('--with-MPI-libs="'+libs+'"')

    # tell hypre configure not to look for blas/lapack [and not use hypre-internal blas]
    args.append('--with-blas-lib="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('--with-lapack-lib=" "')
    args.append('--with-blas=no')
    args.append('--with-lapack=no')
    if self.openmp.found:
      args.append('--with-openmp')

    # explicitly tell hypre BLAS/LAPACK mangling since it may not match Fortran mangling
    if self.blasLapack.mangling == 'underscore':
      mang = 'one-underscore'
    elif self.blasLapack.mangling == 'caps':
      mang = 'caps-no-underscores'
    else:
      mang = 'no-underscores'
    args.append('--with-fmangle-blas='+mang)
    args.append('--with-fmangle-lapack='+mang)

    args.append('--without-mli')
    args.append('--without-fei')
    args.append('--without-superlu')
    if self.indexTypes.integerSize == 64:
      args.append('--enable-bigint')
    # hypre configure assumes the AR flags are passed in with AR
    args = [arg for arg in args if not arg.startswith('AR')]
    args.append('AR="'+self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+'"')
    return [arg for arg in args if not arg in ['--enable-shared']]

  def consistencyChecks(self):
    config.package.GNUPackage.consistencyChecks(self)
    if self.argDB['with-'+self.package]:
      if not self.blasLapack.checkForRoutine('dgels'):
        raise RuntimeError('hypre requires the LAPACK routine dgels(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it')
      self.log.write('Found dgels() in Lapack library as needed by hypre\n')
    return
