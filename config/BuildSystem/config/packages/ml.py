import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.download          = ['git://https://bitbucket.org/petsc/pkg-ml.git','http://ftp.mcs.anl.gov/pub/petsc/externalpackages/ml-6.2-p3.tar.gz']
    self.gitcommit         = 'v6.2-p3'
    self.functions         = ['ML_Set_PrintLevel']
    self.includes          = ['ml_include.h']
    self.liblist           = [['libml.a']]
    self.license           = 'http://trilinos.sandia.gov/'
    self.fc                = 0
    self.double            = 1
    self.complex           = 0
    self.downloadonWindows = 1
    self.requires32bitint  = 1;  # ml uses a combination of "global" indices that can be 64 bit and local indices that are always int therefore it is
                                 # essentially impossible to use ML's 64 bit integer mode with PETSc's --with-64-bit-indices
    self.needsMath         = 1   # ml test needs the system math library
    self.hastests          = 1
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.languages  = framework.require('PETSc.options.languages',   self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]
    return

  def generateLibList(self,dir):
    import os
    '''Normally the one in package.py is used, but ML requires the extra C++ library'''
    libs = ['libml']
    alllibs = []
    for l in libs:
      alllibs.append(l+'.a')
    # Now specify -L ml-lib-path only to the first library
    alllibs[0] = os.path.join(dir,alllibs[0])
    import config.setCompilers
    if self.languages.clanguage == 'C':
      alllibs.extend(self.compilers.cxxlibs)
    return [alllibs]

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--disable-ml-epetra')
    args.append('--disable-ml-aztecoo')
    args.append('--disable-ml-examples')
    args.append('--disable-tests')

    self.framework.pushLanguage('C')
    args.append('--with-cflags="'+self.removeWarningFlags(self.framework.getCompilerFlags())+' -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX '+ self.headers.toStringNoDupes(self.mpi.include)+'"')
    args.append('CPPFLAGS="'+self.headers.toStringNoDupes(self.mpi.include)+'"')
    self.framework.popLanguage()

    if hasattr(self.compilers, 'FC'):
      self.framework.pushLanguage('FC')
      args.append('--with-fflags="'+self.framework.getCompilerFlags()+' '+ self.headers.toStringNoDupes(self.mpi.include)+'"')
      self.framework.popLanguage()
    else:
      args.append('F77=""')

    if hasattr(self.compilers, 'CXX'):
      self.framework.pushLanguage('Cxx')
      args.append('--with-cxxflags="'+self.removeWarningFlags(self.framework.getCompilerFlags())+' -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX '+ self.headers.toStringNoDupes(self.mpi.include)+'"')
      self.framework.popLanguage()
    else:
      raise RuntimeError('Error: ML requires C++ compiler. None specified')

    # ML does not have --with-mpi-include - so specify includes with cflags,fflags,cxxflags,CPPFLAGS
    args.append('--enable-mpi')
    args.append('--with-mpi-libs="'+self.libraries.toString(self.mpi.lib)+'"')
    args.append('--with-blas="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    return args

  def consistencyChecks(self):
    config.package.GNUPackage.consistencyChecks(self)
    if self.argDB['with-'+self.package]:
      # ML requires LAPACK routine dgels() ?
      if not self.blasLapack.checkForRoutine('dgels'):
        raise RuntimeError('ML requires the LAPACK routine dgels(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it')
      if not self.blasLapack.checkForRoutine('dsteqr'):
        raise RuntimeError('ML requires the LAPACK routine dsteqr(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it')
      self.log.write('Found dsteqr() in Lapack library as needed by ML\n')
      self.log.write('Found dgels() in Lapack library as needed by ML\n')

