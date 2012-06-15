import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    #self.download   = ['http://elemental.googlecode.com/files/elemental-0.74.tgz']
    self.download = ['/home/xzhou/temp/elemental-dev-061312.tgz']
    self.liblist    = [['libelemental.a','libplcg.a','libpmrrr.a']]
    #self.functions  = ['GaussianElimination']
    self.includes   = ['elemental.hpp']
    self.cxx              = 1
    self.requires32bitint = 0
    self.complex          = 1
    self.worksonWindows   = 0
    self.downloadonWindows= 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.cmake           = framework.require('PETSc.packages.cmake',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]
    return

  def Install(self):
    import os
    if not self.cmake.found:
      raise RuntimeError('CMake 2.8.5 or above is needed to build Elemental')
    #print 'self.installDir = ', self.installDir
    #print 'self.packageDir = ', self.packageDir
    args = ['-DCMAKE_INSTALL_PREFIX='+self.installDir]
    args.append('-DCMAKE_VERBOSE_MAKEFILE=1')

    self.framework.pushLanguage('C')
    args.append('-DMPI_C_COMPILER="'+self.framework.getCompiler()+'"')
    cflags = self.setCompilers.getCompilerFlags()
    #args.append('-DCMAKE_C_FLAGS:STRING="'+cflags+'"')
    self.framework.popLanguage()

    self.framework.pushLanguage('Cxx')
    args.append('-DMPI_CXX_COMPILER="'+self.framework.getCompiler()+'"')
    cxxflags = self.setCompilers.getCompilerFlags()
    #args.append('-DCMAKE_CXX_FLAGS:STRING="'+cxxflags+'"')
    self.framework.popLanguage()


    """self.framework.pushLanguage('C')
    args.append('-DCMAKE_C_COMPILER="'+self.framework.getCompiler()+'"')
    cflags = self.setCompilers.getCompilerFlags()
    args.append('-DCMAKE_C_FLAGS:STRING="'+cflags+'"')
    self.framework.popLanguage()

    self.framework.pushLanguage('Cxx')

    args.append('-DCMAKE_CXX_COMPILER="'+self.framework.getCompiler()+'"')
    cxxflags = self.setCompilers.getCompilerFlags()
    args.append('-DCMAKE_CXX_FLAGS:STRING="'+cxxflags+'"')
    self.framework.popLanguage()

    if hasattr(self.compilers, 'FC'):
      self.framework.pushLanguage('FC')
      args.append('-DCMAKE_Fortran_COMPILER="'+self.framework.getCompiler()+'"')
      fcflags = self.setCompilers.getCompilerFlags()
      args.append('-DCMAKE_Fortran_FLAGS:STRING="'+fcflags+'"')
      self.framework.popLanguage()
    """
    #if self.sharedLibraries.useShared:
      #args.append('-DSHARED=1')

    #if self.compilerFlags.debugging:
      #args.append('-DDEBUG=1')

    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'elemental'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('elemental'):
      # effectively, this is 'make clean'
      folder = os.path.join(self.packageDir, self.arch)
      #print 'folder = ', folder
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
