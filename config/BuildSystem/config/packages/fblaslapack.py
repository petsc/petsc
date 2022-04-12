import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit              = 'v3.4.2-p3'
    self.download               = ['git://https://bitbucket.org/petsc/pkg-fblaslapack','https://bitbucket.org/petsc/pkg-fblaslapack/get/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames       = ['petsc-pkg-fblaslapack']
    self.precisions             = ['single','double']
    self.downloadonWindows      = 1
    self.skippackagewithoptions = 1
    self.buildLanguages         = ['FC']

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    return

  def configureLibrary(self):
    if self.argDB['with-64-bit-blas-indices']:
      raise RuntimeError('fblaslapack does not support -with-64-bit-blas-indices')
    if hasattr(self.argDB,'known-64-bit-blas-indices') and self.argDB['known-64-bit-blas-indices']:
      raise RuntimeError('fblaslapack does not support -known-64-bit-blas-indices')
    config.package.Package.configureLibrary(self)

  def Install(self):
    import os

    libdir = self.libDir
    confdir = self.confDir
    blasDir = self.packageDir

    g = open(os.path.join(blasDir,'tmpmakefile'),'w')
    f = open(os.path.join(blasDir,'makefile'),'r')
    line = f.readline()
    while line:
      if line.startswith('CC  '):
        cc = self.compilers.CC
        line = 'CC = '+cc+'\n'
      if line.startswith('COPTFLAGS '):
        self.pushLanguage('C')
        line = 'COPTFLAGS  = '+self.getCompilerFlags()
        noopt = self.checkNoOptFlag()
        self.popLanguage()
      if line.startswith('CNOOPT'):
        self.pushLanguage('C')
        line = 'CNOOPT = '+noopt+ ' '+self.getSharedFlag(self.getCompilerFlags())+' '+self.getPointerSizeFlag(self.getCompilerFlags())+' '+self.getWindowsNonOptFlags(self.getCompilerFlags())
        self.popLanguage()
      if line.startswith('FC  '):
        fc = self.compilers.FC
        if fc.find('f90') >= 0 or fc.find('f95') >=0:
          if config.setCompilers.Configure.isIBM(fc, self.log):
            fc = os.path.join(os.path.dirname(fc),'xlf')
            self.log.write('Using IBM f90 compiler, switching to xlf for compiling BLAS/LAPACK\n')
        line = 'FC = '+fc+'\n'
      if line.startswith('FOPTFLAGS '):
        self.pushLanguage('FC')
        line = 'FOPTFLAGS  = '+self.updatePackageFFlags(self.getCompilerFlags())+'\n'
        noopt = self.checkNoOptFlag()
        self.popLanguage()
      if line.startswith('FNOOPT'):
        self.pushLanguage('FC')
        line = 'FNOOPT = '+noopt+' '+self.getSharedFlag(self.getCompilerFlags())+' '+self.getPointerSizeFlag(self.getCompilerFlags())+' '+self.getWindowsNonOptFlags(self.getCompilerFlags())+' '+self.updatePackageFFlags('')+'\n'
        self.popLanguage()
      if line.startswith('AR  '):
        line = 'AR      = '+self.setCompilers.AR+'\n'
      if line.startswith('AR_FLAGS  '):
        line = 'AR_FLAGS      = '+self.setCompilers.AR_FLAGS+'\n'
      if line.startswith('LIB_SUFFIX '):
        line = 'LIB_SUFFIX = '+self.setCompilers.AR_LIB_SUFFIX+'\n'
      if line.startswith('RANLIB  '):
        line = 'RANLIB = '+self.setCompilers.RANLIB+'\n'
      if line.startswith('RM  '):
        line = 'RM = '+self.programs.RM+'\n'
      if line.startswith('OMAKE  '):
        line = 'OMAKE = '+self.make.make+'\n'

      if line.startswith('include'):
        line = '\n'
      if line.find("-no-prec-div") >= 1:
         raise RuntimeError('Some versions of the Intel compiler generate incorrect code on fblaslapack with the option -no-prec-div\nRun configure without this option')
      g.write(line)
      line = f.readline()
    f.close()
    g.close()

    if not self.installNeeded('tmpmakefile'): return self.installDir

    try:
      self.logPrintBox('Compiling FBLASLAPACK; this may take several minutes')
      output1,err1,ret  = config.package.Package.executeShellCommand('cd '+blasDir+' && rm -rf */*.c && make -f tmpmakefile cleanblaslapck cleanlib && '+self.make.make_jnp+' -f tmpmakefile', timeout=2500, log = self.log)
    except RuntimeError as e:
      self.logPrint('Error running make on '+blasDir+': '+str(e))
      raise RuntimeError('Error running make on '+blasDir)
    try:
      output2,err2,ret  = config.package.Package.executeShellCommand('cd '+blasDir+' && mkdir -p '+libdir+' && cp -f libfblas.'+self.setCompilers.AR_LIB_SUFFIX+' libflapack.'+self.setCompilers.AR_LIB_SUFFIX+' '+ libdir, timeout=300, log = self.log)
    except RuntimeError as e:
      self.logPrint('Error moving '+blasDir+' libraries: '+str(e))
      raise RuntimeError('Error moving '+blasDir+' libraries')
    self.postInstall(output1+err1+output2+err2,'tmpmakefile')
    return self.installDir


