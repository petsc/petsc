import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    #self.gitcommit        = '5dc20f1424206f2a09b001e2585fe5c794e60dbf'
    #self.giturls          = ['https://github.com/elemental/Elemental']
    #self.download         = ['http://libelemental.org/pub/releases/Elemental-0.85.tgz']
    self.download         = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/Elemental-0.85-p1.tar.gz']
    self.liblist          = [['libEl.a','libpmrrr.a']]
    self.includes         = ['El.hpp']
    self.cxx              = 1
    self.requirescxx11    = 1
    self.downloadonWindows= 0
    self.hastests         = 1
    self.downloadfilename = 'Elemental'
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.deps            = [self.mpi,self.blasLapack]
    #
    # also requires the ./configure option --with-cxx-dialect=C++11
    return

  def formCMakeConfigureArgs(self):
    if not self.cmake.found:
      raise RuntimeError('CMake > 2.5 is needed to build Elemental')
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DMATH_LIBS:STRING="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    args.append('-DEL_USE_QT5=OFF') # otherwise we would need Qt5 include paths to compile
    args.append('-DBUILD_KISSFFT=OFF')
    args.append('-DBUILD_METIS=OFF')
    args.append('-DBUILD_PARMETIS=OFF')
    args.append('-DINSTALL_PYTHON_PACKAGE=FALSE')
    if self.setCompilers.isDarwin(self.log):
      # shared library build doesn't work on Apple
      args.append('-DBUILD_SHARED_LIBS=off')
    if not self.sharedLibraries.useShared:
      args.append('-DBUILD_SHARED_LIBS=off')

    self.framework.pushLanguage('C')
    args.append('-DMPI_C_COMPILER="'+self.framework.getCompiler()+'"')
    if self.argDB['with-64-bit-indices']:
      args.append('-DEL_USE_64BIT_INTS=ON')
    self.framework.popLanguage()

    self.framework.pushLanguage('Cxx')
    if config.setCompilers.Configure.isSolaris(self.log):
       raise RuntimeError('Sorry, Elemental does not compile with Oracle/Solaris/Sun compilers')
    args.append('-DMPI_CXX_COMPILER="'+self.framework.getCompiler()+'"')
    self.framework.popLanguage()

    if hasattr(self.compilers, 'FC'):
      self.framework.pushLanguage('FC')
      args.append('-DMPI_Fortran_COMPILER="'+self.framework.getCompiler()+'"')
      self.framework.popLanguage()
    return args
