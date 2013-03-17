import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download          = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/metis-5.0.2-p3.tar.gz']
    self.functions         = ['METIS_PartGraphKway']
    self.includes          = ['metis.h']
    self.liblist           = [['libmetis.a']]
    self.needsMath         = 1
    self.complex           = 1
    self.worksonWindows    = 1
    self.downloadonWindows = 1
    self.double            = 0
    self.requires32bitint  = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.cmake           = framework.require('PETSc.packages.cmake',self)
    self.sharedLibraries = framework.require('PETSc.utilities.sharedLibraries', self)
    self.scalartypes     = framework.require('PETSc.utilities.scalarTypes', self)
    self.deps = []
    return

  def Install(self):
    import os
    import shlex

    if not self.cmake.found:
      raise RuntimeError('CMake > 2.5 is needed to build METIS')

    args = ['-DCMAKE_INSTALL_PREFIX='+self.installDir]
    args.append('-DCMAKE_VERBOSE_MAKEFILE=1')
    args.append('-DGKLIB_PATH=../GKlib') # assumes that the 'build' folder is only one directory down

    self.framework.pushLanguage('C')
    args.append('-DCMAKE_C_COMPILER="'+self.framework.getCompiler()+'"')
    args.append('-DCMAKE_AR='+self.setCompilers.AR)
    ranlib = shlex.split(self.setCompilers.RANLIB)[0]
    args.append('-DCMAKE_RANLIB='+ranlib)
    cflags = self.setCompilers.getCompilerFlags()
    args.append('-DCMAKE_C_FLAGS:STRING="'+cflags+'"')
    self.framework.popLanguage()

    if self.sharedLibraries.useShared:
      args.append('-DSHARED=1')

    if self.compilerFlags.debugging:
      args.append('-DDEBUG=1')

    if self.libraryOptions.integerSize == 64:
      args.append('-DMETIS_USE_LONGINDEX=1')

    if self.scalartypes.precision == 'double':
      args.append('-DMETIS_USE_DOUBLEPRECISION=1')
    elif self.scalartypes.precision == 'quad':
      raise RuntimeError('METIS cannot be built with quad precision')

    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'metis'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('metis'):

      # effectively, this is 'make clean'
      folder = os.path.join(self.packageDir, self.arch)
      if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)
        os.mkdir(folder)

      try:
        self.logPrintBox('Configuring METIS; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+folder+' && '+self.cmake.cmake+' .. '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on METIS: '+str(e))
      try:
        self.logPrintBox('Compiling METIS; this may take several minutes')
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+folder+' && make && make install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on METIS: '+str(e))
      self.postInstall(output1+err1+output2+err2,'metis')
    return self.installDir
