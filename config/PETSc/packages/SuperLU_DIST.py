import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download   = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/SuperLU_DIST_2.4-hg-v2.tar.gz']
    self.functions  = ['set_default_options_dist']
    self.includes   = ['superlu_ddefs.h']
    self.liblist    = [['libsuperlu_dist_2.4.a']]
    #
    #  SuperLU_dist supports 64 bit integers but uses ParMetis which does not, it has
    #  a hack that uses the 32 bit parmetis
    #  SuperLU_dist's support for 64 bit integers is nonsense! (Fortran code -qintsize=8 compile options)
    self.requires32bitint = 1;
    self.complex    = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)    
    self.parmetis   = framework.require('PETSc.packages.ParMetis',self)
    self.deps       = [self.mpi,self.blasLapack,self.parmetis]
    return

  def Install(self):
    import os

    g = open(os.path.join(self.packageDir,'make.inc'),'w')
    g.write('DSuperLUroot = '+self.packageDir+'\n')
    g.write('DSUPERLULIB  = $(DSuperLUroot)/libsuperlu_dist_2.4.a\n')
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
    if self.blasLapack.mangling == 'underscore':
      g.write('CDEFS   = -DAdd_')
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
      # set fortran name mangling
      # this mangling information is for both BLAS and the Fortran compiler so cannot use the BlasLapack mangling flag      
      self.setCompilers.popLanguage()
    g.write('NOOPTS       = '+self.blasLapack.getSharedFlag(self.setCompilers.getCompilerFlags())+' '+self.blasLapack.getPrecisionFlag(self.setCompilers.getCompilerFlags())+' '+self.blasLapack.getWindowsNonOptFlags(self.setCompilers.getCompilerFlags())+'\n')
    g.close()

    if self.installNeeded('make.inc'):
      try:
        self.logPrintBox('Compiling superlu_dist; this may take several minutes')
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+';SUPERLU_DIST_INSTALL_DIR='+self.installDir+'/lib;export SUPERLU_DIST_INSTALL_DIR; make clean; make lib LAAUX=""; mv -f *.a '+os.path.join(self.installDir,'lib')+'; cp -f SRC/*.h '+os.path.join(self.installDir,'include')+'/.', timeout=2500, log = self.framework.log)
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
