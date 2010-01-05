import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download  = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/spai_3.0-mar-06.tar.gz']
    self.functions = ['bspai']
    self.includes  = ['spai.h']
    self.liblist   = [['libspai.a']]
    # SPAI include files are in the lib directory
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]
    return

  def Install(self):
    import os

    self.framework.pushLanguage('C')
    if self.blasLapack.mangling == 'underscore':   FTNOPT = ''
    elif self.blasLapack.mangling == 'caps': FTNOPT = ''
    else:                                          FTNOPT = '-DSP2'
    
    args = 'CC = '+self.framework.getCompiler()+'\nCFLAGS = -DMPI '+FTNOPT+' '+self.framework.getCompilerFlags()+' '+self.headers.toString(self.mpi.include)+'\n'
    args = args+'AR         = '+self.setCompilers.AR+'\n'
    args = args+'ARFLAGS    = '+self.setCompilers.AR_FLAGS+'\n'
                                  
    fd = file(os.path.join(self.packageDir,'lib','Makefile.in'),'w')
    fd.write(args)
    self.framework.popLanguage()
    fd.close()

    if self.installNeeded('Makefile.in'):
      self.logPrintBox('Configuring and compiling Spai; this may take several minutes')
      output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+os.path.join(self.packageDir,'lib')+'; make clean; make ; mv -f libspai.a '+os.path.join(self.installDir,'lib','libspai.a'),timeout=250, log = self.framework.log)
      output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+os.path.join(self.packageDir,'lib')+'; cp -f *.h '+os.path.join(self.installDir,'include'),timeout=250, log = self.framework.log)
      try:
        output3,err3,ret3  = PETSc.package.NewPackage.executeShellCommand(self.setCompilers.RANLIB+' '+os.path.join(self.installDir,'lib')+'/libspai.a', timeout=250, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running ranlib on SPAI libraries: '+str(e))
      self.postInstall(output1+err1+output2+err2+output3+err3,'Makefile.in')
    return self.installDir

  def consistencyChecks(self):
    PETSc.package.NewPackage.consistencyChecks(self)
    if self.framework.argDB['with-'+self.package]:
      # SPAI requires dormqr() LAPACK routine
      if not self.blasLapack.checkForRoutine('dormqr'): 
        raise RuntimeError('SPAI requires the LAPACK routine dormqr(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it\nTry using --download-f-blas-lapack=1 option \nIf you are using the IBM ESSL library, it does not contain this function.')
      self.framework.log.write('Found dormqr() in Lapack library as needed by SPAI\n')
    return
