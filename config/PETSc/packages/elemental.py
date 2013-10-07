import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.gitcommit  = '89e7ddc52db68dc5a3f2635733293d7349dbd518'
    self.giturls    = ['https://github.com/elemental/Elemental']
    self.download   = ['http://libelemental.org/pub/releases/elemental-0.81.tgz',
                       'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/elemental-0.81.tgz']
    self.liblist    = [['libelemental.a','libpmrrr.a']]
    self.includes   = ['elemental.hpp']
    self.cxx              = 1
    self.requirescxx11    = 1
    self.requires32bitint = 0
    self.complex          = 1
    self.worksonWindows   = 0
    self.downloadonWindows= 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.cmake           = framework.require('PETSc.packages.cmake',self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.deps            = [self.mpi,self.cmake,self.blasLapack]
    return

  def Install(self):
    import os
    import shlex

    if not self.cmake.found:
      raise RuntimeError('CMake 2.8.5 or above is needed to build Elemental')
    args = ['-DCMAKE_INSTALL_PREFIX='+self.installDir]
    args.append('-DCMAKE_VERBOSE_MAKEFILE=1')
    args.append('-DMATH_LIBS:STRING="'+self.libraries.toString(self.blasLapack.dlib)+'"')

    self.framework.pushLanguage('C')
    args.append('-DMPI_C_COMPILER="'+self.framework.getCompiler()+'"')
    args.append('-DCMAKE_C_COMPILER="'+self.framework.getCompiler()+'"')
    args.append('-DCMAKE_AR='+self.setCompilers.AR)
    ranlib = shlex.split(self.setCompilers.RANLIB)[0]
    args.append('-DCMAKE_RANLIB='+ranlib)
    cflags = self.setCompilers.getCompilerFlags()
    args.append('-DCMAKE_C_FLAGS:STRING="'+cflags+'"')
    if self.framework.argDB['with-64-bit-indices']:
      args.append('-DUSE_64BIT_INTS=ON')
    self.framework.popLanguage()

    self.framework.pushLanguage('Cxx')
    import config.setCompilers
    if config.setCompilers.Configure.isSun(self.framework.getCompiler()):
      raise RuntimeError('Sorry, Elemental does not compile with Oracle/Solaris/Sun compilers')
    args.append('-DMPI_CXX_COMPILER="'+self.framework.getCompiler()+'"')
    args.append('-DCMAKE_CXX_COMPILER="'+self.framework.getCompiler()+'"')
    cxxflags = self.framework.getCompilerFlags()
    args.append('-DCMAKE_CXX_FLAGS:STRING="'+cxxflags+'"')
    self.framework.popLanguage()

    if hasattr(self.compilers, 'FC'):
      self.framework.pushLanguage('FC')
      args.append('-DCMAKE_Fortran_COMPILER="'+self.framework.getCompiler()+'"')
      fcflags = self.framework.getCompilerFlags()
      args.append('-DCMAKE_Fortran_FLAGS:STRING="'+fcflags+'"')
      self.framework.popLanguage()

    """if self.sharedLibraries.useShared:
      args.append('-DSHARE_LIBRARIES=ON')

    if self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=PureDebug')"""

    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'elemental'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('elemental'):
      # effectively, this is 'make clean'
      folder = os.path.join(self.packageDir, self.arch)
      if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)
        os.mkdir(folder)
      try:
        self.logPrintBox('Configuring Elemental; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+folder+' && '+self.cmake.cmake+' .. '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on Elemental: '+str(e))
      try:
        self.logPrintBox('Compiling Elemental; this may take several minutes')
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+folder+' && make && make install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on Elemental: '+str(e))
      self.postInstall(output1+err1+output2+err2,'elemental')
    return self.installDir
