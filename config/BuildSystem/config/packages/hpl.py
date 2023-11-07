import config.package

# only install dmatgen from HPL, not the entire hpl package
class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.GNUPackage.__init__(self, framework)
    self.version          = '2.3.0'
    self.maxversion       = '2.5.100000'
    self.versionname      = 'PACKAGE_VERSION'
    self.download         = ['https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz']
    self.downloaddirnames = ['hpl']
    self.functions        = ['HPL_dmatgen']
    self.includes         = ['hpl.h']
    self.liblist          = [['libhpl.a']]
    self.complex          = 0
    self.precisions       = ['double']

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.mathlib    = framework.require('config.packages.mathlib',self)
    self.deps       = [self.mpi,self.blasLapack,self.mathlib]

  def Install(self):
    import os, sys
    import config.base

    libDir         = os.path.join(self.installDir, 'lib')
    includeDir     = os.path.join(self.installDir, 'include')
    srcDir         = os.path.join(self.packageDir,'testing','matgen')
    makefile       = os.path.join(srcDir, 'makefile')

    g = open(makefile,'w')
    g.write('SHELL            = '+self.programs.SHELL+'\n')
    g.write('CP               = '+self.programs.cp+'\n')
    g.write('RM               = '+self.programs.RM+'\n')
    g.write('MKDIR            = '+self.programs.mkdir+'\n')
    g.write('OMAKE            = '+self.make.make+' '+self.make.noprintdirflag+'\n')

    g.write('CLINKER          = '+self.getLinker()+'\n')
    g.write('AR               = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS          = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('AR_LIB_SUFFIX    = '+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('RANLIB           = '+self.setCompilers.RANLIB+'\n')
    g.write('SL_LINKER_SUFFIX = '+self.setCompilers.sharedLibraryExt+'\n')

    g.write('PREFIX           = '+self.installDir+'\n')
    g.write('LIBDIR           = '+libDir+'\n')
    g.write('INSTALL_LIB_DIR  = '+libDir+'\n')
    g.write('HPLLIB           = libhpl.$(AR_LIB_SUFFIX)\n')
    g.write('SHLIB            = libhpl\n')

    self.pushLanguage('C')
    cflags = self.updatePackageCFlags(self.getCompilerFlags())
    cflags += ' '+self.headers.toString('.')
    cflags += ' -fPIC'

    g.write('CC               = '+self.getCompiler()+'\n')
    g.write('CFLAGS           = '+cflags+'\n')
    self.popLanguage()
    g.write('clean:\n')
    g.write('	${RM} -f *.o\n')
    g.write('$(HPLLIB):\n')
    g.write('	$(CC) -I../../include $(CFLAGS)  -c *.c\n')
    g.write('	$(AR) $(ARFLAGS) $(HPLLIB) *.o\n')
    g.close()

    # compile & install
    if self.installNeeded(makefile):
      try:
        self.logPrintBox('Compiling HPL dmatgen; this may take several minutes')
        output1,err1,ret1  = config.package.Package.executeShellCommand('cd '+srcDir+' && make clean && make libhpl.'+self.setCompilers.AR_LIB_SUFFIX+' && make clean', timeout=250, log = self.log)
      except RuntimeError as e:
        raise RuntimeError('Error running make on HPL dmatgen: '+str(e))
      self.logPrintBox('Installing HPL dmatgen; this may take several minutes')
      output,err,ret = config.package.Package.executeShellCommand('mkdir -p '+os.path.join(self.installDir,'lib'), timeout=25, log=self.log)
      output,err,ret = config.package.Package.executeShellCommand('mkdir -p '+os.path.join(self.installDir,'include'), timeout=25, log=self.log)
      output2,err2,ret2  = config.package.Package.executeShellCommand('cp -f '+os.path.join(srcDir,'libhpl.'+self.setCompilers.AR_LIB_SUFFIX)+' '+os.path.join(self.installDir,'lib'), timeout=60, log = self.log)
      output2,err2,ret2  = config.package.Package.executeShellCommand('cp -f '+os.path.join(self.packageDir, 'include', '*.h')+' '+includeDir, timeout=60, log = self.log)
      self.postInstall(output1+err1+output2+err2,makefile)
    return self.installDir

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    return
