import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download         = ['https://web.cels.anl.gov/projects/petsc/download/externalpackages/spai-3.0-p1.tar.gz']
    self.functions        = ['bspai']
    self.includes         = ['spai.h']
    self.liblist          = [['libspai.a']]
    # SPAI include files are in the lib directory
    self.precisions       = ['double']
    self.requires32bitint = 1
    self.complex          = 0
    self.hastests         = 1
    self.requirekandr     = 1
    self.brokengnu23      = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.deps       = [self.mpi,self.blasLapack]
    return

  def Install(self):
    import os

    self.pushLanguage('C')
    if self.blasLapack.mangling == 'underscore':   FTNOPT = ''
    elif self.blasLapack.mangling == 'caps': FTNOPT = ''
    else:                                          FTNOPT = '-DSP2'

    args = 'CC = '+self.getCompiler()+'\nCFLAGS = -DSPAI_USE_MPI '+FTNOPT+' '+self.updatePackageCFlags(self.getCompilerFlags())+' '+self.headers.toString(self.mpi.include)+'\n'
    args = args+'AR         = '+self.setCompilers.AR+'\n'
    args = args+'ARFLAGS    = '+self.setCompilers.AR_FLAGS+'\n'

    fd = open(os.path.join(self.packageDir,'lib','Makefile.in'),'w')
    fd.write(args)
    self.popLanguage()
    fd.close()

    if self.installNeeded('Makefile.in'):
      self.logPrintBox('Configuring, compiling and installing Spai; this may take several minutes')
      config.package.Package.executeShellCommand('mkdir -p '+os.path.join(self.installDir,'lib'), timeout=2500, log=self.log)
      config.package.Package.executeShellCommand('cd '+os.path.join(self.packageDir,'lib')+' && make clean && make && cp -f libspai.a '+os.path.join(self.installDir,'lib','libspai.a'),timeout=250, log = self.log)
      config.package.Package.executeShellCommand('cd '+os.path.join(self.packageDir,'lib')+' && cp -f *.h '+os.path.join(self.installDir,'include'),timeout=250, log = self.log)
      try:
        config.package.Package.executeShellCommand(self.setCompilers.RANLIB+' '+os.path.join(self.installDir,'lib')+'/libspai.a', timeout=250, log = self.log)
      except RuntimeError as e:
        raise RuntimeError('Error running ranlib on SPAI libraries: '+str(e))
      self.postInstall('Makefile.in')
    return self.installDir

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if self.argDB['with-'+self.package]:
      # SPAI requires dormqr() LAPACK routine
      if not self.blasLapack.checkForRoutine('dormqr'):
        raise RuntimeError('SPAI requires the LAPACK routine dormqr(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it\nTry using --download-fblaslapack=1 option \nIf you are using the IBM ESSL library, it does not contain this function.')
      self.log.write('Found dormqr() in Lapack library as needed by SPAI\n')
    return
