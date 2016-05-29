import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download         = ['http://www.netlib.org/scalapack/scalapack-2.0.2.tgz',
                             'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/scalapack-2.0.2.tgz']
    self.includes         = []
    self.liblist          = [['libscalapack.a']]
    self.functions        = ['pssytrd']
    self.functionsFortran = 1
    self.fc               = 1
    self.useddirectly     = 0 # PETSc does not use ScaLAPACK, it is only used by MUMPS
    self.downloadonWindows= 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.deps       = [self.mpi, self.blasLapack]
    return

  # this code should be removed and a proper dependency on flibs should be somehow added to setupDependencies()
  def generateLibList(self,dir):
    '''scalapack can require -lgfortran when using f2cblaslapack'''
    alllibs = config.package.Package.generateLibList(self,dir)
    for a in alllibs[:]:
      b=a[:]
      b.extend(self.compilers.flibs)
      alllibs.append(b)
    return alllibs

  def Install(self):
    import os
    g = open(os.path.join(self.packageDir,'SLmake.inc'),'w')
    g.write('SCALAPACKLIB = '+'libscalapack.'+self.setCompilers.AR_LIB_SUFFIX+' \n')
    g.write('LIBS         = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('MPIINC       = '+self.headers.toString(self.mpi.include)+'\n')
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
    g.write('CCFLAGS      = '+self.removeWarningFlags(self.setCompilers.getCompilerFlags())+' $(MPIINC)\n')
    g.write('CCLOADER     = '+self.setCompilers.getLinker()+'\n')
    g.write('CCLOADFLAGS  = '+self.setCompilers.getLinkerFlags()+'\n')
    self.setCompilers.popLanguage()
    g.write('ARCH         = '+self.setCompilers.AR+'\n')
    g.write('ARCHFLAGS    = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    g.close()

    if self.installNeeded('SLmake.inc'):
      try:
        output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && make cleanlib', timeout=2500, log = self.log)
      except RuntimeError, e:
        pass
      try:
        self.logPrintBox('Compiling and installing Scalapack; this may take several minutes')
        self.installDirProvider.printSudoPasswordMessage()
        libDir = os.path.join(self.installDir, self.libdir)
        output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && make lib && '+self.installSudo+'mkdir -p '+libDir+' && '+self.installSudo+'cp libscalapack.* '+libDir, timeout=2500, log = self.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on SCALAPACK: '+str(e))
      self.postInstall(output,'SLmake.inc')
    return self.installDir
