import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download               = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/f2cblaslapack-3.4.2.q4.tar.gz']
    self.downloadonWindows      = 1
    self.skippackagewithoptions = 1
    self.installwithbatch       = 1

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.blis = framework.require('config.packages.blis', self)
    self.odeps = [self.blis]
    return

  def configureLibrary(self):
    if self.argDB['with-64-bit-blas-indices']:
      raise RuntimeError('f2cblaslapack does not support -with-64-bit-blas-indices')
    if hasattr(self.argDB,'known-64-bit-blas-indices') and self.argDB['known-64-bit-blas-indices']:
      raise RuntimeError('f2cblaslapack does not support -known-64-bit-blas-indices')
    config.package.Package.configureLibrary(self)

  def Install(self):
    import os

    if self.defaultPrecision == '__float128': make_target = 'blas_qlib lapack_qlib'
    elif self.defaultPrecision == '__fp16': make_target   = 'blas_hlib lapack_hlib'
    elif self.blis.found: make_target = 'blasaux_lib lapack_lib'
    else: make_target = 'blas_lib lapack_lib'

    libdir = self.libDir
    confdir = self.confDir

    with open(os.path.join(self.packageDir,'tmpmakefile'),'w') as g:
      with open(os.path.join(self.packageDir,'makefile'),'r') as f:
        for line in f:
          if line.startswith('CC  '):
            cc = self.compilers.CC
            line = 'CC = '+cc+'\n'
          if line.startswith('COPTFLAGS '):
            self.pushLanguage('C')
            line = 'COPTFLAGS  = '+self.updatePackageCFlags(self.getCompilerFlags())+'\n'
            self.popLanguage()
          if line.startswith('CNOOPT'):
            self.pushLanguage('C')
            noopt = self.checkNoOptFlag()
            line = 'CNOOPT = '+noopt+ ' '+self.getSharedFlag(self.getCompilerFlags())+' '+self.getPointerSizeFlag(self.getCompilerFlags())+' '+self.getWindowsNonOptFlags(self.getCompilerFlags())+'\n'
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
            line = 'RM = '+self.programs.RM+'\nMAKE = '+self.make.make+'\n'
          if line.startswith('include'):
            line = '\n'
          if line.find("-no-prec-div") >= 0:
             raise RuntimeError('Some versions of the Intel compiler generate incorrect code on f2cblaslapack with the option -no-prec-div\nRun configure without this option')
          g.write(line)
        otherlibs = '''
blas_hlib:\n\
\t-@cd blas;   $(MAKE) hlib $(MAKE_OPTIONS_BLAS)\n\
\t-@$(RANLIB) $(BLAS_LIB_NAME)\n\
lapack_hlib:\n\
\t-@cd lapack; $(MAKE) hlib $(MAKE_OPTIONS_LAPACK)\n\
\t-@$(RANLIB) $(LAPACK_LIB_NAME)\n\
blas_qlib:\n\
\t-@cd blas;   $(MAKE) qlib $(MAKE_OPTIONS_BLAS)\n\
\t-@$(RANLIB) $(BLAS_LIB_NAME)\n\
lapack_qlib:\n\
\t-@cd lapack; $(MAKE) qlib $(MAKE_OPTIONS_LAPACK)\n\
\t-@$(RANLIB) $(LAPACK_LIB_NAME)\n'''
      g.write(otherlibs)

    if not self.installNeeded('tmpmakefile'): return self.installDir

    try:
      self.logPrintBox('Compiling F2CBLASLAPACK; this may take several minutes')
      output1,err1,ret  = config.package.Package.executeShellCommandSeq([
        self.make.make_jnp_list + ['-f', 'tmpmakefile', 'cleanblaslapck', 'cleanlib'],
        self.make.make_jnp_list + ['-f', 'tmpmakefile'] + make_target.split(),
        ], cwd=self.packageDir, timeout=2500, log = self.log)
    except RuntimeError as e:
      self.logPrint('Error running make on '+self.packageDir+': '+str(e))
      raise RuntimeError('Error running make on '+self.packageDir)
    try:
      self.logPrintBox('Installing F2CBLASLAPACK')
      self.installDirProvider.printSudoPasswordMessage()
      output2,err2,ret  = config.package.Package.executeShellCommandSeq([
        self.withSudo('mkdir', '-p', libdir),
        self.withSudo('cp', '-f', 'libf2clapack.' + self.setCompilers.AR_LIB_SUFFIX, 'libf2cblas.' + self.setCompilers.AR_LIB_SUFFIX, libdir),
        ], cwd=self.packageDir, timeout=60, log = self.log)
    except RuntimeError as e:
      self.logPrint('Error moving '+self.packageDir+' libraries: '+str(e))
      raise RuntimeError('Error moving '+self.packageDir+' libraries')
    self.postInstall(output1+err1+output2+err2,'tmpmakefile')
    return self.installDir

