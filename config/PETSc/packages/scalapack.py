import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download         = ['http://www.netlib.org/scalapack/scalapack-2.0.2.tgz',
                             'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/scalapack-2.0.2.tgz']
    self.includes         = []
    self.liblist          = [[],['libscalapack.a']]
    self.functions        = ['pssytrd']
    self.requires32bitint = 0
    self.functionsFortran = 1
    self.double           = 0
    self.complex          = 1
    self.useddirectly     = 0 # PETSc does not use ScaLAPACK, it is only used by MUMPS
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
    if not hasattr(self.setCompilers, 'FC'):
      raise RuntimeError('SCALAPACK requires Fortran for automatic installation')

    g = open(os.path.join(self.packageDir,'SLmake.inc'),'w')
    g.write('SCALAPACKLIB = '+'libscalapack.'+self.setCompilers.AR_LIB_SUFFIX+' \n')
    g.write('LIBS         = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    # this mangling information is for both BLAS and the Fortran compiler so cannot use the BlasLapack mangling flag
    if self.compilers.fortranManglingDoubleUnderscore:
      blah = 'f77IsF2C'
    elif self.compilers.fortranMangling == 'underscore':
      blah = 'Add_'
    elif self.compilers.fortranMangling == 'caps':
      blah = 'UpCase'
    else:
      blah = 'NoChange'
    g.write('CDEFS        =-D'+blah+'\n')
    self.setCompilers.pushLanguage('FC')
    g.write('FC           = '+self.setCompilers.getCompiler()+'\n')
    g.write('FCFLAGS      = '+self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','').replace('-Mfree','')+'\n')
    g.write('FCLOADER     = '+self.setCompilers.getLinker()+'\n')
    g.write('FCLOADFLAGS  = '+self.setCompilers.getLinkerFlags()+'\n')
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
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make cleanlib', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        pass
      try:
        self.logPrintBox('Compiling Scalapack; this may take several minutes')
        libDir = os.path.join(self.installDir, self.libdir)
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make lib && mv libscalapack.* '+libDir, timeout=2500, log = self.framework.log)
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
        raise RuntimeError('SCALAPACK requires a COMPLETE BLAS and LAPACK, it cannot be used with the --download-f2cblaslapack\nUse --download-f-blas-lapack option instead.')
    return
