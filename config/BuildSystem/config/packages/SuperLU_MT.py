import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download   = ['http://crd-legacy.lbl.gov/~xiaoye/SuperLU/superlu_mt_2.1.tar.gz']
    self.functions  = ['cCreate_CompCol_Matrix']
    self.includes   = ['pcsp_defs.h']
    self.liblist    = [['libsuperlu_mt_OPENMP.a']]
    # SuperLU_Dist does not work with --download-fblaslapack with Compaqf90 compiler on windows.
    # However it should work with intel ifort.
    self.worksonWindows   = 1
    self.downloadonWindows= 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.indexTypes     = framework.require('PETSc.options.indexTypes', self)
    self.blasLapack     = framework.require('config.packages.BlasLapack',self)
    self.deps           = [self.blasLapack]
    return

  def Install(self):
    import os

    g = open(os.path.join(self.packageDir,'make.inc'),'w')
    g.write('TMGLIB       = libtmglib_OPENMP.'+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('SUPERLULIB   = libsuperlu_mt_OPENMP.'+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('BLASDEF      = -DUSE_VENDOR_BLAS\n')
    g.write('BLASLIB      = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('LIBS         = $(DSUPERLULIB) $(BLASLIB)\n')
    g.write('ARCH         = '+self.setCompilers.AR+'\n')
    g.write('ARCHFLAGS    = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS       = '+self.setCompilers.getCompilerFlags()+'-D_OPENMP\n')
    g.write('LOADER       = '+self.setCompilers.getLinker()+'\n')
    g.write('LOADOPTS     = \n')
    self.setCompilers.popLanguage()
    # set blas/lapack name mangling
    if self.blasLapack.mangling == 'underscore':
      g.write('CDEFS   = -DAdd_')
    elif self.blasLapack.mangling == 'caps':
      g.write('CDEFS   = -DUpCase')
    else:
      g.write('CDEFS   = -DNoChange')
    if self.indexTypes.integerSize == 64:
      g.write(' -D_LONGINT')
    g.write('\n')
    # not sure what this is for
    g.write('NOOPTS       = '+self.getSharedFlag(self.setCompilers.getCompilerFlags())+' '+self.getPointerSizeFlag(self.setCompilers.getCompilerFlags())+' '+self.getWindowsNonOptFlags(self.setCompilers.getCompilerFlags())+'\n')
    g.close()

    if self.installNeeded('make.inc'):
      try:
        self.logPrintBox('Compiling and installing superlu_mt; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+os.path.join(self.installDir,'lib'), timeout=2500, log=self.log)
        output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+os.path.join(self.installDir,'include'), timeout=2500, log=self.log)
        if not os.path.exists(os.path.join(self.packageDir,'lib')):
          os.makedirs(os.path.join(self.packageDir,'lib'))
        output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && make clean && make all LAAUX="" && '+self.installSudo+'cp -f lib/*.'+self.setCompilers.AR_LIB_SUFFIX+' '+os.path.join(self.installDir,self.libdir,'')+' && '+self.installSudo+' cp -f SRC/*.h '+os.path.join(self.installDir,self.includedir,''), timeout=2500, log = self.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on SUPERLU_MT: '+str(e))
      self.postInstall(output+err,'make.inc')
    return self.installDir

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if self.argDB['with-'+self.package]:
      if not self.blasLapack.checkForRoutine('slamch'):
        raise RuntimeError('SuperLU_MT requires the BLAS routine slamch()')
      self.log.write('Found slamch() in BLAS library as needed by SuperLU_DIST\n')
      if not self.blasLapack.checkForRoutine('dlamch'):
        raise RuntimeError('SuperLU_MT requires the BLAS routine dlamch()')
      self.log.write('Found dlamch() in BLAS library as needed by SuperLU_DIST\n')
      if not self.blasLapack.checkForRoutine('xerbla'):
        raise RuntimeError('SuperLU_MT requires the BLAS routine xerbla()')
      self.log.write('Found xerbla() in BLAS library as needed by SuperLU_DIST\n')
    return
