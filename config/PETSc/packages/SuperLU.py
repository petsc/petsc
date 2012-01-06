import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download     = ['http://crd.lbl.gov/~xiaoye/SuperLU/superlu_4.3.tar.gz']
    self.functions    = ['set_default_options']
    self.includes     = ['slu_ddefs.h']
    self.liblist      = [['libsuperlu_4.3.a']]
    self.complex      = 1
    # SuperLU has NO support for 64 bit integers, use SuperLU_Dist if you need that
    self.excludedDirs = ['SuperLU_DIST']
    # SuperLU does not work with --download-f-blas-lapack with Compaqf90 compiler on windows.
    # However it should work with intel ifort.
    self.worksonWindows   = 1
    self.downloadonWindows= 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = self.framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.blasLapack]
    return

  def Install(self):
    import os
    # Get the SUPERLU directories

    g = open(os.path.join(self.packageDir,'make.inc'),'w')
    g.write('SuperLUroot  = '+self.packageDir+'\n')
    g.write('TMGLIB       = tmglib.'+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('SUPERLULIB   = $(SuperLUroot)/lib/libsuperlu_4.3.'+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('BLASLIB      = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('BLASDEF      = -DUSE_VENDOR_BLAS\n')
    g.write('LIBS         = $(SUPERLULIB) $(BLASLIB)\n')
    g.write('ARCH         = '+self.setCompilers.AR+'\n')
    g.write('ARCHFLAGS    = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS       = '+self.setCompilers.getCompilerFlags()+'\n')
    g.write('LOADER       = '+self.setCompilers.getLinker()+'\n') 
    g.write('LOADOPTS     = \n') 
    self.setCompilers.popLanguage()

    # set blas name mangling
    if self.blasLapack.mangling == 'underscore':
      g.write('CDEFS        = -DAdd_')
    elif self.blasLapack.mangling == 'caps':
      g.write('CDEFS   = -DUpCase')
    else:
      g.write('CDEFS   = -DNoChange')
    g.write('\n')

    g.write('MATLAB       =\n')
    g.write('NOOPTS       = '+self.blasLapack.getSharedFlag(self.setCompilers.getCompilerFlags())+' '+self.blasLapack.getPrecisionFlag(self.setCompilers.getCompilerFlags())+' '+self.blasLapack.getWindowsNonOptFlags(self.setCompilers.getCompilerFlags())+'\n')
    g.close()
    if self.installNeeded('make.inc'):
      try:
        self.logPrintBox('Compiling superlu; this may take several minutes')
        if not os.path.exists(os.path.join(self.packageDir,'lib')):
          os.makedirs(os.path.join(self.packageDir,'lib'))
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make clean && make superlulib LAAUX="" SLASRC="" DLASRC="" CLASRC="" ZLASRC="" SCLAUX="" DZLAUX="" && cp -f lib/*.'+self.setCompilers.AR_LIB_SUFFIX+' '+os.path.join(self.installDir,self.libdir,'')+' &&  cp -f SRC/*.h '+os.path.join(self.installDir,self.includedir,''), timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on SUPERLU: '+str(e))
      self.postInstall(output+err,'make.inc')
    return self.installDir

  def consistencyChecks(self):
    PETSc.package.NewPackage.consistencyChecks(self)
    if self.framework.argDB['with-'+self.package]:
      if not self.blasLapack.checkForRoutine('slamch'): 
        raise RuntimeError('SuperLU requires the LAPACK routine slamch()')
      self.framework.log.write('Found slamch() in Lapack library as needed by SuperLU\n')

      if not self.blasLapack.checkForRoutine('dlamch'): 
        raise RuntimeError('SuperLU requires the LAPACK routine dlamch()')
      self.framework.log.write('Found dlamch() in Lapack library as needed by SuperLU\n')

      if not self.blasLapack.checkForRoutine('xerbla'): 
        raise RuntimeError('SuperLU requires the BLAS routine xerbla()')
      self.framework.log.write('Found xerbla() in BLAS library as needed by SuperLU\n')
    return
