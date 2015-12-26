import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit        = 'v4.2'
    self.download         = ['git://https://bitbucket.org/petsc/pkg-superlu_dist.git',
                             'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/superlu_dist_4.2.tar.gz']
    self.functions        = ['set_default_options_dist']
    self.includes         = ['superlu_ddefs.h']
    self.liblist          = [['libsuperlu_dist_4.2.a']]
    # SuperLU_Dist does not work with --download-fblaslapack with Compaqf90 compiler on windows.
    # However it should work with intel ifort.
    self.downloadonWindows= 1
    self.hastests         = 1
    self.hastestsdatafiles= 1
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('SUPERLU_DIST', '-download-superlu_dist-gpu=<bool>',    nargs.ArgBool(None, 0, 'Install Superlu_DIST to use GPUs'))

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.indexTypes     = framework.require('PETSc.options.indexTypes', self)
    self.blasLapack     = framework.require('config.packages.BlasLapack',self)
    self.metis          = framework.require('config.packages.metis',self)
    self.parmetis       = framework.require('config.packages.parmetis',self)
    self.mpi            = framework.require('config.packages.MPI',self)
    if self.framework.argDB['download-superlu_dist-gpu']:
      self.cuda           = framework.require('config.packages.cuda',self)
      self.openmp         = framework.require('config.packages.openmp',self)
      self.deps           = [self.mpi,self.blasLapack,self.parmetis,self.metis,self.cuda,self.openmp]
    else:
      self.deps           = [self.mpi,self.blasLapack,self.parmetis,self.metis]
    return

  def Install(self):
    import os
    if (self.compilers.c99flag == None):
      raise RuntimeError('SUPERLU_DIST: install requires c99 compiler. Configure cold not determine compatilbe compiler flag. Perhaps you can specify via CFLAG')
    if not self.make.haveGNUMake:
      raise RuntimeError('SUPERLU_DIST: install requires GNU make. Suggest using --download-make')

    g = open(os.path.join(self.packageDir,'make.inc'),'w')
    g.write('DSuperLUroot = '+self.packageDir+'\n')
    g.write('DSUPERLULIB  = $(DSuperLUroot)/libsuperlu_dist_4.2.'+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('BLASDEF      = -DUSE_VENDOR_BLAS\n')
    g.write('BLASLIB      = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('INCS         = '+self.headers.toString(self.mpi.include)+' '+self.headers.toString(self.parmetis.include)+' '+self.headers.toString(self.metis.include)+'\n')
    g.write('MPILIB       = '+self.libraries.toString(self.mpi.lib)+'\n')
    g.write('PMETISLIB    = '+self.libraries.toString(self.parmetis.lib)+'\n')
    g.write('METISLIB     = '+self.libraries.toString(self.metis.lib)+'\n')

    if self.framework.argDB['download-superlu_dist-gpu']:
      g.write('ACC          = GPU\n')
      g.write('CUDAFLAGS    = -DGPU_ACC '+self.headers.toString(self.cuda.include)+'\n')
      g.write('CUDALIB      = '+self.libraries.toString(self.cuda.lib)+'\n')
    else:
      g.write('ACC          = \n')
      g.write('CUDAFLAGS    = \n')
      g.write('CUDALIB      = \n')

    g.write('LIBS         = $(DSUPERLULIB) $(PMETISLIB) $(METISLIB) $(BLASLIB) $(MPILIB) $(CUDALIB)\n')
    g.write('ARCH         = '+self.setCompilers.AR+'\n')
    g.write('ARCHFLAGS    = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS       = $(INCS) $(CUDAFLAGS) '+self.setCompilers.getCompilerFlags()+' '+self.compilers.c99flag+'\n')
    g.write('LOADER       = '+self.setCompilers.getLinker()+' '+'\n')
    g.write('LOADOPTS     = \n')
    self.setCompilers.popLanguage()
    # set blas/lapack name mangling
    if self.blasLapack.mangling == 'underscore':
      g.write('CDEFS        = -DAdd_')
    elif self.blasLapack.mangling == 'caps':
      g.write('CDEFS        = -DUpCase')
    else:
      g.write('CDEFS        = -DNoChange')
    if self.indexTypes.integerSize == 64:
      g.write(' -D_LONGINT')
    g.write('\n')
    g.write('NOOPTS       = '+self.getSharedFlag(self.setCompilers.getCompilerFlags())+' '+self.getPointerSizeFlag(self.setCompilers.getCompilerFlags())+' '+self.getWindowsNonOptFlags(self.setCompilers.getCompilerFlags())+'\n')
    g.close()

    if self.installNeeded('make.inc'):
      try:
        self.logPrintBox('Compiling and installing superlu_dist; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()

        if not os.path.exists(os.path.join(self.packageDir,'lib')):
          os.makedirs(os.path.join(self.packageDir,'lib'))
        output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && '+self.make.make+' clean && '+self.make.make+' lib LAAUX="smach.o dmach.o" && '+self.installSudo+'cp -f *.'+self.setCompilers.AR_LIB_SUFFIX+' '+os.path.join(self.installDir,self.libdir,'')+' && '+self.installSudo+'cp -f SRC/*.h '+os.path.join(self.installDir,self.includedir,''), timeout=2500, log = self.log)
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
