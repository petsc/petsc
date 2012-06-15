import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download   = ['http://elemental.googlecode.com/files/elemental-0.74.tgz']
    self.liblist    = [['libelemental.a','libplcg.a','libpmrrr.a']]
    #self.functions  = ['elemental']
    self.includes   = ['elemental.hpp']
    self.requires32bitint = 0
    self.complex          = 1
    self.worksonWindows   = 0
    self.downloadonWindows= 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    #self.parmetis   = framework.require('PETSc.packages.parmetis',self)
    self.deps       = [self.mpi,self.blasLapack]
    return

  def Install(self):
    import os
    g = open(os.path.join(self.packageDir,'make.inc'),'w')
    g.write('elementalRoot = '+self.packageDir+'\n')
    g.write('elementalLIB  = $(elementalRoot)/libsu0.'+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('BLASDEF      = -DUSE_VENDOR_BLAS\n')
    g.write('BLASLIB      = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('IMPI         = '+self.headers.toString(self.mpi.include)+'\n')
    g.write('MPILIB       = '+self.libraries.toString(self.mpi.lib)+'\n')
    #g.write('PMETISLIB    = '+self.libraries.toString(self.parmetis.lib)+'\n')
    g.write('LIBS         = $(BLASLIB) $(MPILIB)\n')
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
      g.write('CDEFS        = -DAdd_')
    elif self.blasLapack.mangling == 'caps':
      g.write('CDEFS        = -DUpCase')
    else:
      g.write('CDEFS        = -DNoChange')
    if self.libraryOptions.integerSize == 64:
      g.write(' -D_LONGINT')
    g.write('\n')
    # not sure what this is for
    g.write('NOOPTS       = '+self.blasLapack.getSharedFlag(self.setCompilers.getCompilerFlags())+' '+self.blasLapack.getPrecisionFlag(self.setCompilers.getCompilerFlags())+' '+self.blasLapack.getWindowsNonOptFlags(self.setCompilers.getCompilerFlags())+'\n')
    g.close()

    if self.installNeeded('make.inc'):
      try:
        self.logPrintBox('Compiling elemental; this may take several minutes')
        if not os.path.exists(os.path.join(self.packageDir,'PureRelease')):
          os.makedirs(os.path.join(self.packageDir,'PureRelease'))
        libDir     = os.path.join(self.installDir, self.libdir)
        includeDir = os.path.join(self.installDir, self.includedir)
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'/PureRelease && cmake -D CMAKE_INSTALL_PREFIX='+self.packageDir+'/PureRelease .. && make && make install && cp -f lib/*.* '+libDir+' && cp -rf include/* '+includeDir, timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on elemental: '+str(e))
      self.postInstall(output+err,'make.inc')
    return self.installDir

  def consistencyChecks(self):
    """PETSc.package.NewPackage.consistencyChecks(self)
    if self.framework.argDB['with-'+self.package]:
      if not self.blasLapack.checkForRoutine('slamch'):
        raise RuntimeError('SuperLU_DIST requires the BLAS routine slamch()')
      self.framework.log.write('Found slamch() in BLAS library as needed by SuperLU_DIST\n')
      if not self.blasLapack.checkForRoutine('dlamch'):
        raise RuntimeError('SuperLU_DIST requires the BLAS routine dlamch()')
      self.framework.log.write('Found dlamch() in BLAS library as needed by SuperLU_DIST\n')
      if not self.blasLapack.checkForRoutine('xerbla'):
        raise RuntimeError('SuperLU_DIST requires the BLAS routine xerbla()')
      self.framework.log.write('Found xerbla() in BLAS library as needed by SuperLU_DIST\n')"""
    return
