import config.package
import os

class Configure(config.package.GNUPackage):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.gitcommit         = 'v6.2-p4'
    self.download          = ['git://https://bitbucket.org/petsc/pkg-ml.git','https://bitbucket.org/petsc/pkg-ml/get/'+self.gitcommit+'.tar.gz']
    self.functions         = ['ML_Set_PrintLevel']
    self.includes          = ['ml_include.h']
    self.liblist           = [['libml.a']]
    self.license           = 'http://trilinos.sandia.gov/'
    self.cxx               = 1
    self.precisions        = ['double']
    self.complex           = 0
    self.downloadonWindows = 1
    self.requires32bitint  = 1;  # ml uses a combination of "global" indices that can be 64 bit and local indices that are always int therefore it is
                                 # essentially impossible to use ML's 64 bit integer mode with PETSc's --with-64-bit-indices
    self.hastests          = 1
    self.downloaddirnames  = ['petsc-pkg-ml']
    return

  def setupDependencies(self, framework):
    config.package.GNUPackage.setupDependencies(self, framework)
    self.cxxlibs    = framework.require('config.packages.cxxlibs',self)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.mathlib    = framework.require('config.packages.mathlib',self)
    self.deps       = [self.mpi,self.blasLapack,self.cxxlibs,self.mathlib]
    return

  def formGNUConfigureArgs(self):
    args = config.package.GNUPackage.formGNUConfigureArgs(self)
    args.append('--disable-ml-epetra')
    args.append('--disable-ml-aztecoo')
    args.append('--disable-ml-examples')
    args.append('--disable-tests')
    args.append('--enable-libcheck')

    self.pushLanguage('C')
    args.append('--with-cflags="'+self.updatePackageCFlags(self.getCompilerFlags())+' -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX '+ self.headers.toStringNoDupes(self.mpi.include)+'"')
    args.append('CPPFLAGS="'+self.headers.toStringNoDupes(self.mpi.include)+'"')
    self.popLanguage()

    if hasattr(self.compilers, 'FC'):
      self.pushLanguage('FC')
      args.append('--with-fflags="'+self.getCompilerFlags()+' '+ self.headers.toStringNoDupes(self.mpi.include)+'"')
      self.popLanguage()
    else:
      args.append('F77=""')
    self.pushLanguage('Cxx')
    args.append('--with-cxxflags="'+self.updatePackageCxxFlags(self.getCompilerFlags())+' -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX '+ self.headers.toStringNoDupes(self.mpi.include)+'"')
    self.popLanguage()

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

