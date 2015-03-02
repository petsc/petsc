import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit  = '749f33d8104157767d443ff1a1d151642751486d' # v3.3 @ 2013-04-10
    self.giturls    = ['https://bitbucket.org/petsc/pkg-superlu_dist.git']
    self.download   = ['http://crd-legacy.lbl.gov/~xiaoye/SuperLU/superlu_dist_3.3.tar.gz']
    self.functions  = ['set_default_options_dist']
    self.includes   = ['superlu_ddefs.h']
    self.liblist    = [['libsuperlu_dist_3.3.a']]
    # SuperLU_Dist does not work with --download-fblaslapack with Compaqf90 compiler on windows.
    # However it should work with intel ifort.
    self.downloadonWindows= 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.indexTypes     = framework.require('PETSc.options.indexTypes', self)
    self.blasLapack     = framework.require('config.packages.BlasLapack',self)
    self.parmetis       = framework.require('config.packages.parmetis',self)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.deps           = [self.mpi,self.blasLapack,self.parmetis]
    return

  def Install(self):
    import os

    g = open(os.path.join(self.packageDir,'make.inc'),'w')
    g.write('DSuperLUroot = '+self.packageDir+'\n')
    g.write('DSUPERLULIB  = $(DSuperLUroot)/libsuperlu_dist_3.3.'+self.setCompilers.AR_LIB_SUFFIX+'\n')
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
    if self.indexTypes.integerSize == 64:
      g.write(' -D_LONGINT')
    g.write('\n')
    # not sure what this is for
    g.write('NOOPTS       = '+self.getSharedFlag(self.setCompilers.getCompilerFlags())+' '+self.getPointerSizeFlag(self.setCompilers.getCompilerFlags())+' '+self.getWindowsNonOptFlags(self.setCompilers.getCompilerFlags())+'\n')
    g.close()

    if self.installNeeded('make.inc'):
      try:
        self.logPrintBox('Compiling and installing superlu_dist; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()

        if not os.path.exists(os.path.join(self.packageDir,'lib')):
          os.makedirs(os.path.join(self.packageDir,'lib'))
        output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && make clean && make lib LAAUX="" && '+self.installSudo+'cp -f *.'+self.setCompilers.AR_LIB_SUFFIX+' '+os.path.join(self.installDir,self.libdir,'')+' && '+self.installSudo+'cp -f SRC/*.h '+os.path.join(self.installDir,self.includedir,''), timeout=2500, log = self.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on SUPERLU_DIST: '+str(e))
      self.postInstall(output+err,'make.inc')
    return self.installDir

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if self.argDB['with-'+self.package]:
      if not self.blasLapack.checkForRoutine('slamch'):
        raise RuntimeError('SuperLU_DIST requires the BLAS routine slamch()')
      self.log.write('Found slamch() in BLAS library as needed by SuperLU_DIST\n')
      if not self.blasLapack.checkForRoutine('dlamch'):
        raise RuntimeError('SuperLU_DIST requires the BLAS routine dlamch()')
      self.log.write('Found dlamch() in BLAS library as needed by SuperLU_DIST\n')
      if not self.blasLapack.checkForRoutine('xerbla'):
        raise RuntimeError('SuperLU_DIST requires the BLAS routine xerbla()')
      self.log.write('Found xerbla() in BLAS library as needed by SuperLU_DIST\n')
    return
