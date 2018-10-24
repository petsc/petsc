import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download   = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/spai-3.0-p1.tar.gz']
    self.functions  = ['bspai']
    self.includes   = ['spai.h']
    self.liblist    = [['libspai.a']]
    # SPAI include files are in the lib directory
    self.precisions = ['double']
    self.complex    = 0
    self.hastests   = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.deps       = [self.mpi,self.blasLapack]
    return

  def Install(self):
    import os

    self.framework.pushLanguage('C')
    if self.blasLapack.mangling == 'underscore':   FTNOPT = ''
    elif self.blasLapack.mangling == 'caps': FTNOPT = ''
    else:                                          FTNOPT = '-DSP2'

    args = 'CC = '+self.framework.getCompiler()+'\nCFLAGS = -DSPAI_USE_MPI '+FTNOPT+' '+self.removeWarningFlags(self.framework.getCompilerFlags())+' '+self.headers.toString(self.mpi.include)+'\n'
    args = args+'AR         = '+self.setCompilers.AR+'\n'
    args = args+'ARFLAGS    = '+self.setCompilers.AR_FLAGS+'\n'

    fd = open(os.path.join(self.packageDir,'lib','Makefile.in'),'w')
    fd.write(args)
    self.framework.popLanguage()
    fd.close()

    if self.installNeeded('Makefile.in'):
      self.logPrintBox('Configuring, compiling and installing Spai; this may take several minutes')
      self.installDirProvider.printSudoPasswordMessage()
      output,err,ret = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+os.path.join(self.installDir,'lib'), timeout=2500, log=self.log)
      output1,err1,ret1  = config.package.Package.executeShellCommand('cd '+os.path.join(self.packageDir,'lib')+' && make clean && make && '+self.installSudo+' cp -f libspai.a '+os.path.join(self.installDir,'lib','libspai.a'),timeout=250, log = self.log)
      output2,err2,ret2  = config.package.Package.executeShellCommand('cd '+os.path.join(self.packageDir,'lib')+' && '+self.installSudo+' cp -f *.h '+os.path.join(self.installDir,'include'),timeout=250, log = self.log)
      try:
        output3,err3,ret3  = config.package.Package.executeShellCommand(self.installSudo+self.setCompilers.RANLIB+' '+os.path.join(self.installDir,'lib')+'/libspai.a', timeout=250, log = self.log)
      except RuntimeError as e:
        raise RuntimeError('Error running ranlib on SPAI libraries: '+str(e))
      self.postInstall(output1+err1+output2+err2+output3+err3,'Makefile.in')
    return self.installDir

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if self.argDB['with-'+self.package]:
      # SPAI requires dormqr() LAPACK routine
      if not self.blasLapack.checkForRoutine('dormqr'):
        raise RuntimeError('SPAI requires the LAPACK routine dormqr(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it\nTry using --download-fblaslapack=1 option \nIf you are using the IBM ESSL library, it does not contain this function.')
      self.log.write('Found dormqr() in Lapack library as needed by SPAI\n')
    return
