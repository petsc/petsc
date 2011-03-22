import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download  = ['ftp://ftp.mcs.anl.gov/pub/mpi/mpe/mpe2.tar.gz']
    self.functions = ['MPE_Log_event']
    self.includes  = ['mpe.h']
    self.liblist   = [['libmpe.a']]
    self.complex   = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps = [self.mpi]
    return
          
  def Install(self):
    import os

    args = ['--prefix='+self.installDir]
    
    self.framework.pushLanguage('C')
    args.append('CFLAGS="'+self.framework.getCompilerFlags()+'"')
    args.append('MPI_CFLAGS="'+self.framework.getCompilerFlags()+'"')
    args.append('CC="'+self.framework.getCompiler()+'"')
    self.framework.popLanguage()

    args.append('--disable-f77')
    args.append('MPI_INC="'+self.headers.toString(self.mpi.include)+'"')
    args.append('MPI_LIBS="'+self.libraries.toStringNoDupes(self.mpi.lib)+'"')

    args = ' '.join(args)
    
    fd = file(os.path.join(self.packageDir,'mpe'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('mpe'):
      try:
        self.logPrintBox('Configuring mpe; this may take several minutes')
        output  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=2000, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on MPE: '+str(e))
      # Build MPE
      try:
        self.logPrintBox('Compiling mpe; this may take several minutes')
        output  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make clean && make && make install', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on MPE: '+str(e))
    return self.installDir
