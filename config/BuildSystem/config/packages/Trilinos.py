import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = 'trilinos-release-11-14-branch'
    self.giturls          = ['https://software.sandia.gov/trilinos/repositories/publicTrilinos']
    self.download         = ['none']
    self.liblist          = [['libelemental.a','libpmrrr.a']]
    self.includes         = ['elemental.hpp']
    self.cxx              = 1
    self.requirescxx11    = 1
    self.downloadonWindows= 0
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.deps            = [self.mpi,self.blasLapack]
    #
    # also requires the ./configure option --with-cxx-dialect=C++11
    return

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)

    args.append('-DTPL_ENABLE_MPI=ON')
    #    args.append('-DTrilinos_ENABLE_ALL_PACKAGES=ON')
    args.append('-DTrilinos_ENABLE_Epetra=ON')
    args.append('-DTrilinos_ENABLE_AztecOO=ON')
    args.append('-DTrilinos_ENABLE_Ifpack=ON')

    # The documentation specifically says:
    #     WARNING: Do not try to hack the system and set:
    #     TPL_BLAS_LIBRARIES:PATH="-L/some/dir -llib1 -llib2 ..."
    #     This is not compatible with proper CMake usage and it not guaranteed to be supported.
    # We do it anyways because the precribed way of providing the BLAS/LAPACK libraries is insane.
    args.append('-DTPL_BLAS_LIBRARIES="'+self.libraries.toString(self.blasLapack.dlib)+'"')

    # From the docs at http://trilinos.org/download/public-git-repository/
    #TIP: The arguments passed to CMake when building from the Trilinos public repository must include
    args.append('-DTrilinos_ASSERT_MISSING_PACKAGES=OFF')

    self.framework.pushLanguage('C')
    args.append('-DMPI_C_COMPILER="'+self.framework.getCompiler()+'"')
    self.framework.popLanguage()

    self.framework.pushLanguage('Cxx')
    args.append('-DMPI_CXX_COMPILER="'+self.framework.getCompiler()+'"')
    self.framework.popLanguage()

    if hasattr(self.setCompilers, 'FC'):
      self.framework.pushLanguage('FC')
      args.append('-DMPI_Fortran_COMPILER="'+self.framework.getCompiler()+'"')
      self.framework.popLanguage()
    else:
      args.append('-DTrilinos_ENABLE_Fortran=OFF')

    return args




