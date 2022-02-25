import config.package

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = '962cf01ce0ccb5cca6d6bb81e9b1d0b46cac9592' # petsc/0.87.7 on May 31, 2021
    self.download         = ['git://https://github.com/elemental/Elemental']
    self.liblist          = [['libEl.a','libElSuiteSparse.a','libpmrrr.a']]
    self.includes         = ['El.hpp']
    self.precisions       = ['single','double']
    self.buildLanguages   = ['Cxx']
    self.maxCxxVersion    = 'c++14'
    self.downloadonWindows= 0
    self.hastests         = 1
    self.hastestsdatafiles= 1
    self.downloaddirnames = ['Elemental']
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.sharedLibraries = framework.require('PETSc.options.sharedLibraries', self)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.metis           = framework.require('config.packages.metis',self)
    self.parmetis        = framework.require('config.packages.parmetis',self)
    self.deps            = [self.mpi,self.blasLapack,self.metis,self.parmetis]
    #
    # also requires the ./configure option --with-cxx-dialect=C++11
    return

  def formCMakeConfigureArgs(self):
    if not self.parmetis.ComputeVertexSeparator:
      raise RuntimeError('Elemental requires modified Parmetis! Use options: --download-metis=1 --download-parmetis=1')
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    if self.compilerFlags.debugging:
      args.append('-DEL_ZERO_INIT=ON')
    args.append('-DEL_DISABLE_VALGRIND=ON')
    args.append('-DEL_USE_QT5=OFF') # otherwise we would need Qt5 include paths to compile
    args.append('-DEL_FORCE_METIS_BUILD=OFF')
    args.append('-DEL_FORCE_PARMETIS_BUILD=OFF')
    args.append('-DEL_PREVENT_METIS_DOWNLOAD=ON')
    args.append('-DEL_PREVENT_PARMETIS_DOWNLOAD=ON')
    args.append('-DINSTALL_PYTHON_PACKAGE=FALSE')
    args.append('-DPARMETIS_TEST_RUNS=TRUE')
    args.append('-DEL_DISABLE_SCALAPACK=ON')
    args.append('-DEL_DISABLE_MPFR=ON')

    if self.metis.include:
      args.append('-DMETIS_INCLUDE_DIR:STRING="'+self.metis.include[0]+'"')
    args.append('-DMETIS_LIBRARY:STRING="'+self.libraries.toString(self.metis.lib)+'"')
    if self.parmetis.include:
      args.append('-DPARMETIS_INCLUDE_DIR:STRING="'+self.parmetis.include[0]+'"')
    args.append('-DPARMETIS_LIBRARY:STRING="'+self.libraries.toString(self.parmetis.lib)+'"')
    args.append('-DMATH_LIBS:STRING="'+self.libraries.toString(self.blasLapack.dlib)+'"')
    if self.setCompilers.isDarwin(self.log):
      # shared library build doesn't work on Apple
      args = self.rmArgsStartsWith(args,'-DBUILD_SHARED_LIBS')
      args.append('-DBUILD_SHARED_LIBS=off')

    if self.argDB['with-64-bit-indices']:
      args.append('-DEL_USE_64BIT_INTS=ON')

    self.pushLanguage('Cxx')
    if config.setCompilers.Configure.isSolaris(self.log):
       raise RuntimeError('Elemental does not compile with Oracle/Solaris/Sun compilers')
    self.popLanguage()
    return args
