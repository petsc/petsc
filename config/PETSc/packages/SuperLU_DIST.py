import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download   = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/superlu_dist_3.0-p1.tar.gz']
    self.functions  = ['set_default_options_dist']
    self.includes   = ['superlu_ddefs.h']
    self.liblist    = [['libsuperlu_dist_3.0.a']]
    self.requires32bitint = 0
    self.complex          = 1
    # SuperLU_Dist does not work with --download-f-blas-lapack with Compaqf90 compiler on windows.
    # However it should work with intel ifort.
    self.worksonWindows   = 1
    self.downloadonWindows= 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.parmetis   = framework.require('PETSc.packages.parmetis',self)
    self.deps       = [self.mpi,self.blasLapack,self.parmetis]
    return

  def Install(self):
    import os

    g = open(os.path.join(self.packageDir,'make.inc'),'w')
    g.write('DSuperLUroot = '+self.packageDir+'\n')
    g.write('DSUPERLULIB  = $(DSuperLUroot)/libsuperlu_dist_3.0.'+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('BLASDEF      = -DUSE_VENDOR_BLAS\n')
    g.write('BLASLIB      = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('IMPI         = '+self.headers.toString(self.mpi.include)+'\n')
    g.write('MPILIB       = '+self.libraries.toString(self.mpi.lib)+'\n')
    g.write('PMETISLIB    = '+self.libraries.toString(self.parmetis.lib)+'\n')
    g.write('LIBS         = $(DSUPERLULIB) $(BLASLIB) $(PMETISLIB) $(MPILIB)\n')
    g.write('ARCH         = '+self.setCompilers.AR+'\n')
    g.write('ARCHFLAGS    = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+' $(IMPI)\n') #build fails without $(IMPI)
    g.write('CFLAGS       = '+self.setCompilers.getCompilerFlags()+'\n')
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
    if self.libraryOptions.integerSize == 64:
      g.write(' -D_LONGINT')
    g.write('\n')
    # not sure what this is for
    g.write('NOOPTS       = '+self.blasLapack.getSharedFlag(self.setCompilers.getCompilerFlags())+' '+self.blasLapack.getPrecisionFlag(self.setCompilers.getCompilerFlags())+' '+self.blasLapack.getWindowsNonOptFlags(self.setCompilers.getCompilerFlags())+'\n')
    g.close()

    if self.installNeeded('make.inc'):
      try:
        self.logPrintBox('Compiling superlu_dist; this may take several minutes')
        if not os.path.exists(os.path.join(self.packageDir,'lib')):
          os.makedirs(os.path.join(self.packageDir,'lib'))
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make clean && make lib LAAUX="" && mv -f *.'+self.setCompilers.AR_LIB_SUFFIX+' '+os.path.join(self.installDir,self.libdir,'')+' && cp -f SRC/*.h '+os.path.join(self.installDir,self.includedir,''), timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on SUPERLU_DIST: '+str(e))
      self.postInstall(output+err,'make.inc')
    return self.installDir

  def consistencyChecks(self):
    PETSc.package.NewPackage.consistencyChecks(self)
    if self.framework.argDB['with-'+self.package]:
      if not self.blasLapack.checkForRoutine('slamch'):
        raise RuntimeError('SuperLU_DIST requires the BLAS routine slamch()')
      self.framework.log.write('Found slamch() in BLAS library as needed by SuperLU_DIST\n')
      if not self.blasLapack.checkForRoutine('dlamch'):
        raise RuntimeError('SuperLU_DIST requires the BLAS routine dlamch()')
      self.framework.log.write('Found dlamch() in BLAS library as needed by SuperLU_DIST\n')
      if not self.blasLapack.checkForRoutine('xerbla'):
        raise RuntimeError('SuperLU_DIST requires the BLAS routine xerbla()')
      self.framework.log.write('Found xerbla() in BLAS library as needed by SuperLU_DIST\n')
    return
