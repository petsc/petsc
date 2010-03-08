import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download     = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/superlu_4.0-March_7_2010.tar.gz']
    self.functions    = ['set_default_options']
    self.includes     = ['slu_ddefs.h']
    self.liblist      = [['libsuperlu_4.0.a']]
    self.complex      = 1
    # SuperLU has NO support for 64 bit integers, use SuperLU_Dist if you need that
    self.excludedDirs = ['SuperLU_DIST']
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
    g.write('TMGLIB       = tmglib.a\n')
    g.write('SUPERLULIB   = $(SuperLUroot)/lib/libsuperlu_4.0.a\n')
    g.write('BLASLIB      = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('BLASDEF      = -DUSE_VENDOR_BLAS\n')
    g.write('LIBS	  = $(SUPERLULIB) $(BLASLIB)\n')
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
    if self.framework.argDB['with-64-bit-indices']:
      g.write(' -D_LONGINT')
    g.write('\n')

    if hasattr(self.compilers, 'FC'):
      self.setCompilers.pushLanguage('FC')
      g.write('FORTRAN      = '+self.setCompilers.getCompiler()+'\n')
      g.write('FFLAGS       = '+self.setCompilers.getCompilerFlags().replace('-Mfree','')+'\n')
      self.setCompilers.popLanguage()
    else:
      g.write('FORTRAN    = \n')
      g.write('FFLAGS     = \n')
    g.write('MATLAB       =\n')
    g.write('NOOPTS       = '+self.blasLapack.getSharedFlag(self.setCompilers.getCompilerFlags())+' '+self.blasLapack.getPrecisionFlag(self.setCompilers.getCompilerFlags())+' '+self.blasLapack.getWindowsNonOptFlags(self.setCompilers.getCompilerFlags())+'\n')
    g.close()
    if self.installNeeded('make.inc'):
      try:
        self.logPrintBox('Compiling superlu; this may take several minutes')
        output,err,ret = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'; SUPERLU_INSTALL_DIR='+self.installDir+'/lib; export SUPERLU_INSTALL_DIR; make clean; make lib LAAUX="" SLASRC="" DLASRC="" CLASRC="" ZLASRC="" SCLAUX="" DZLAUX=""; cp -f lib/*.a '+os.path.join(self.installDir,'lib')+';  cp -f SRC/*.h '+os.path.join(self.installDir,self.includedir)+'/.', timeout=2500, log = self.framework.log)
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
