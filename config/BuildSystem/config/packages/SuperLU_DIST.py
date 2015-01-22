import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit  = 'bdddaa1c55e208b48f96d6281b8713b55f836c6e' # v4.0-p1 feb-27-2015
    self.giturls    = ['https://bitbucket.org/petsc/pkg-superlu_dist.git']
    self.download   = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/superlu_dist_4.0-p1.tar.gz']
    self.functions  = ['set_default_options_dist']
    self.includes   = ['superlu_ddefs.h']
    self.liblist    = [['libsuperlu_dist_4.0.a']]
    # SuperLU_Dist does not work with --download-fblaslapack with Compaqf90 compiler on windows.
    # However it should work with intel ifort.
    self.downloadonWindows= 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.indexTypes     = framework.require('PETSc.options.indexTypes', self)
    self.blasLapack     = framework.require('config.packages.BlasLapack',self)
    self.metis          = framework.require('config.packages.metis',self)
    self.parmetis       = framework.require('config.packages.parmetis',self)
    self.mpi            = framework.require('config.packages.MPI',self)
    self.deps           = [self.mpi,self.blasLapack,self.parmetis]
    return

  def Install(self):
    import os
    if (self.compilers.c99flag == None):
      raise RuntimeError('SUPERLU_DIST: install requires c99 compiler. Configure cold not determine compatilbe compiler flag. Perhaps you can specify via CFLAG')
    if not self.make.haveGNUMake:
      raise RuntimeError('SUPERLU_DIST: install requires GNU make. Suggest using --download-make')

    g = open(os.path.join(self.packageDir,'make.inc'),'w')
    g.write('DSuperLUroot = '+self.packageDir+'\n')
    g.write('DSUPERLULIB  = $(DSuperLUroot)/libsuperlu_dist_4.0.'+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('BLASDEF      = -DUSE_VENDOR_BLAS\n')
    g.write('BLASLIB      = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('INCS         = '+self.headers.toString(self.mpi.include)+' '+self.headers.toString(self.parmetis.include)+' '+self.headers.toString(self.metis.include)+'\n')
    g.write('MPILIB       = '+self.libraries.toString(self.mpi.lib)+'\n')
    g.write('PMETISLIB    = '+self.libraries.toString(self.parmetis.lib)+'\n')
    g.write('LIBS         = $(DSUPERLULIB) $(BLASLIB) $(PMETISLIB) $(MPILIB)\n')
    g.write('ARCH         = '+self.setCompilers.AR+'\n')
    g.write('ARCHFLAGS    = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS       = $(INCS) '+self.setCompilers.getCompilerFlags()+''+self.compilers.c99flag+'\n')
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
    g.close()

    if self.installNeeded('make.inc'):
      try:
        self.logPrintBox('Compiling and installing superlu_dist; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()

        if not os.path.exists(os.path.join(self.packageDir,'lib')):
          os.makedirs(os.path.join(self.packageDir,'lib'))
        output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && '+self.make.make+' clean && '+self.make.make+' lib LAAUX="" && '+self.installSudo+'cp -f *.'+self.setCompilers.AR_LIB_SUFFIX+' '+os.path.join(self.installDir,self.libdir,'')+' && '+self.installSudo+'cp -f SRC/*.h '+os.path.join(self.installDir,self.includedir,''), timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on SUPERLU_DIST: '+str(e))
      self.postInstall(output+err,'make.inc')
    return self.installDir

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
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
