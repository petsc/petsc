import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    # use the version from PETSc ftp site - it has lapack removed
    self.download  = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/scalapack.tgz']
    self.includes  = []
    self.liblist   = [[],['libscalapack.a']]
    self.functions = ['pssytrd']
    self.requires32bitint = 0;
    self.functionsFortran = 1
    self.complex   = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.blacs      = framework.require('PETSc.packages.blacs',self)
    self.deps       = [self.blacs, self.mpi, self.blasLapack]
    return

  def Install(self):
    import os
    if not hasattr(self.setCompilers, 'FC'):
      raise RuntimeError('SCALAPACK requires Fortran for automatic installation')

    g = open(os.path.join(self.packageDir,'SLmake.inc'),'w')
    g.write('SHELL        = /bin/sh\n')
    g.write('home         = '+self.getDir()+'\n')
    g.write('USEMPI       = -DUsingMpiBlacs\n')
    g.write('SENDIS       = -DSndIsLocBlk\n')
    if (self.mpi.commf2c):
      g.write('WHATMPI      = -DUseMpi2\n')
    else:
      g.write('WHATMPI      = -DCSAMEF77\n')
    g.write('BLACSDBGLVL  = -DBlacsDebugLvl=1\n')
    g.write('BLACSLIB     = '+self.libraries.toString(self.blacs.lib)+'\n') 
    g.write('SMPLIB       = '+self.libraries.toString(self.mpi.lib)+'\n')
    g.write('SCALAPACKLIB = '+os.path.join(self.installDir,self.libdir,'libscalapack.a')+' \n')
    g.write('CBLACSLIB    = $(BLACSCINIT) $(BLACSLIB) $(BLACSCINIT)\n')
    g.write('FBLACSLIB    = $(BLACSFINIT) $(BLACSLIB) $(BLACSFINIT)\n')
    # this mangling information is for both BLAS and the Fortran compiler so cannot use the BlasLapack mangling flag    
    if self.compilers.fortranManglingDoubleUnderscore:
      blah = 'f77IsF2C'
    elif self.compilers.fortranMangling == 'underscore':
      blah = 'Add_'
    elif self.compilers.fortranMangling == 'caps':
      blah = 'UpCase'
    else:
      blah = 'NoChange'
    g.write('CDEFS        =-D'+blah+' -DUsingMpiBlacs\n')
    g.write('PBLASdir     = $(home)/PBLAS\n')
    g.write('SRCdir       = $(home)/SRC\n')
    g.write('TOOLSdir     = $(home)/TOOLS\n')
    g.write('REDISTdir    = $(home)/REDIST\n')
    self.setCompilers.pushLanguage('FC')  
    g.write('F77          = '+self.setCompilers.getCompiler()+'\n')
    g.write('F77FLAGS     = '+self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','').replace('-Mfree','')+'\n')
    g.write('F77LOADER    = '+self.setCompilers.getLinker()+'\n')      
    g.write('F77LOADFLAGS = '+self.setCompilers.getLinkerFlags()+'\n')
    self.setCompilers.popLanguage()
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n')
    g.write('CCFLAGS      = '+self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','')+'\n')      
    g.write('CCLOADER     = '+self.setCompilers.getLinker()+'\n')
    g.write('CCLOADFLAGS  = '+self.setCompilers.getLinkerFlags()+'\n')
    self.setCompilers.popLanguage()
    g.write('ARCH         = '+self.setCompilers.AR+'\n')
    g.write('ARCHFLAGS    = '+self.setCompilers.AR_FLAGS+'\n')    
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')    
    g.close()

    if self.installNeeded('SLmake.inc'):
      try:
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+';make cleanlib', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        pass
      try:
        self.logPrintBox('Compiling Scalapack; this may take several minutes')
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+';make', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on SCALAPACK: '+str(e))
      self.postInstall(output,'SLmake.inc')
    return self.installDir

  def checkLib(self, lib, func, mangle, otherLibs = []):
    oldLibs = self.compilers.LIBS
    found = self.libraries.check(lib,func, otherLibs = otherLibs+self.mpi.lib+self.blasLapack.lib+self.compilers.flibs,fortranMangle=mangle)
    self.compilers.LIBS = oldLibs
    if found:
      self.framework.log.write('Found function '+str(func)+' in '+str(lib)+'\n')
    return found

  def consistencyChecks(self):
    PETSc.package.NewPackage.consistencyChecks(self)
    if self.framework.argDB['with-'+self.package]:
      # SCALAPACK requires ALL of BLAS/LAPACK
      if self.blasLapack.f2c:
        raise RuntimeError('SCALAPACK requires a COMPLETE BLAS and LAPACK, it cannot be used with the --download-c-blas-lapack\nUse --download-f-blas-lapack option instead.')
    return
