import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download  = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/blacs-dev.tar.gz']
    self.liblist   = [[],['libblacs.a']]
    self.includes  = []
    self.fc        = 1
    self.functions = ['blacs_pinfo']
    self.requires32bitint = 0
    self.functionsFortran = 1
    self.complex   = 1
    self.useddirectly     = 0 # PETSc does not use BLACS, it is only used by ScaLAPACK which is used by MUMPS
    self.worksonWindows   = 1
    self.downloadonWindows= 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.mpi, self.blasLapack]
    return

  def Install(self):
    import os

    g = open(os.path.join(self.packageDir,'Bmake.Inc'),'w')
    g.write('SHELL     = /bin/sh\n')
    g.write('COMMLIB   = MPI\n')
    g.write('SENDIS    = -DSndIsLocBlk\n')
    if (self.mpi.commf2c):
      g.write('WHATMPI      = -DUseMpi2\n')
    else:
      g.write('WHATMPI      = -DCSAMEF77\n')
    g.write('DEBUGLVL  = -DBlacsDebugLvl=1\n')
    g.write('BLACSdir  = '+self.packageDir+'\n')
    g.write('BLACSLIB  = '+os.path.join(self.installDir,self.libdir,'libblacs.'+self.setCompilers.AR_LIB_SUFFIX)+'\n')
    g.write('MPILIB    = '+self.libraries.toString(self.mpi.lib)+'\n')
    g.write('SYSINC    = '+self.headers.toString(self.mpi.include)+'\n')
    g.write('BTLIBS    = $(BLACSLIB)  $(MPILIB) \n')

    # this mangling information is for both BLAS and the Fortran compiler so cannot use the BlasLapack mangling flag
    if self.compilers.fortranManglingDoubleUnderscore:
      blah = 'f77IsF2C'
    elif self.compilers.fortranMangling == 'underscore':
      blah = 'Add_'
    elif self.compilers.fortranMangling == 'caps':
      blah = 'UpCase'
    else:
      blah = 'NoChange'
    g.write('INTFACE   = -D'+blah+'\n')
    g.write('DEFS1     = -DSYSINC $(SYSINC) $(INTFACE) $(DEFBSTOP) $(DEFCOMBTOP) $(DEBUGLVL)\n')
    g.write('BLACSDEFS = $(DEFS1) $(SENDIS) $(BUFF) $(TRANSCOMM) $(WHATMPI) $(SYSERRORS)\n')
    self.setCompilers.pushLanguage('FC')  
    g.write('F77       = '+self.setCompilers.getCompiler()+'\n')
    g.write('F77FLAGS  = '+self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','')+'$(SYSINC)\n')
    g.write('F77LOADER = '+self.setCompilers.getLinker()+'\n')      
    g.write('F77LOADFLAGS ='+self.setCompilers.getLinkerFlags()+'\n')
    self.setCompilers.popLanguage()     
    self.setCompilers.pushLanguage('C')
    g.write('CC          = '+self.setCompilers.getCompiler()+'\n')
    g.write('CCFLAGS     = '+self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','')+'\n')      
    g.write('CCLOADER    = '+self.setCompilers.getLinker()+'\n')
    g.write('CCLOADFLAGS = '+self.setCompilers.getLinkerFlags()+'\n')
    self.setCompilers.popLanguage()
    g.write('ARCH        = '+self.setCompilers.AR+'\n')
    g.write('ARCHFLAGS   = '+self.setCompilers.AR_FLAGS+'\n')    
    g.write('RANLIB      = '+self.setCompilers.RANLIB+'\n')    
    g.close()

    if self.installNeeded('Bmake.Inc'):
      try:
        self.logPrintBox('Compiling Blacs; this may take several minutes')
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+os.path.join(self.packageDir,'SRC','MPI')+' && make clean && make', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on BLACS: '+str(e))
      self.postInstall(output+err,'Bmake.Inc')
    return self.installDir
