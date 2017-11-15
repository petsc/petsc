import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    #self.download   = ['http://ftp.mcs.anl.gov/pub/petsc/tmp/elemental-dev-072512.tar.gz']
    self.download   = ['/home/xzhou/temp/clique.tgz']
    self.liblist    = [['libclique.a','libparmetis-addons.a','libmetis-addons.a']]
    self.includes   = ['clique.hpp']
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
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.elemental       = framework.require('PETSc.packages.elemental',self)
    self.parmetis        = framework.require('PETSc.packages.parmetis',self)
    self.metis           = framework.require('PETSc.packages.metis',self)
    self.deps            = [self.cmake,self.blasLapack,self.elemental,self.parmetis,self.metis]
    return

  def Install(self):
    import os
    import shlex

    args = ['-DCMAKE_INSTALL_PREFIX='+self.installDir]
    args.append('-DCMAKE_VERBOSE_MAKEFILE=1')
    args.append('-DMATH_LIBS:STRING="'+self.libraries.toString(self.blasLapack.dlib)+'"')

    self.framework.pushLanguage('C')
    args.append('-DMPI_C_COMPILER="'+self.framework.getCompiler()+'"')
    cflags = self.setCompilers.getCompilerFlags()
    print 'cflags',cflags
    args.append('-DCMAKE_C_FLAGS:STRING="'+cflags+'"')
    self.framework.popLanguage()

    self.framework.pushLanguage('Cxx')
    args.append('-DMPI_CXX_COMPILER="'+self.framework.getCompiler()+'"')
    args.append('-DCMAKE_AR='+self.setCompilers.AR)
    ranlib = shlex.split(self.setCompilers.RANLIB)[0]
    args.append('-DCMAKE_RANLIB='+ranlib)
    cxxflags = self.setCompilers.getCompilerFlags()
    args.append('-DCMAKE_CXX_FLAGS:STRING="'+cxxflags+'"')
    self.framework.popLanguage()

    if hasattr(self.compilers, 'FC'):
      self.framework.pushLanguage('FC')
      args.append('-DCMAKE_Fortran_COMPILER="'+self.framework.getCompiler()+'"')
      fcflags = self.setCompilers.getCompilerFlags()
      args.append('-DCMAKE_Fortran_FLAGS:STRING="'+fcflags+'"')
      self.framework.popLanguage()

    """if self.sharedLibraries.useShared:
      args.append('-DSHARE_LIBRARIES=ON')

    if self.compilerFlags.debugging:
      args.append('-DCMAKE_BUILD_TYPE=PureDebug')"""

    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'clique'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('clique'):
      # effectively, this is 'make clean'
      folder = os.path.join(self.packageDir, self.arch)
      if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)
        os.mkdir(folder)
      try:
        self.logPrintBox('Configuring Clique; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+folder+' && '+self.cmake.cmake+' .. '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on Clique: '+str(e))
      try:
        self.logPrintBox('Compiling Clique; this may take several minutes')
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+folder+' && make && make install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on Clique: '+str(e))
      self.postInstall(output1+err1+output2+err2,'clique')
    return self.installDir
