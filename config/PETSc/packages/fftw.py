import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    # host locally as fftw.org url can expire after new release.
    self.download  = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/fftw-3.3.2.tar.gz','http://www.fftw.org/fftw-3.3.2.tar.gz']
    self.functions = ['fftw_malloc'] 
    self.includes  = ['fftw3-mpi.h']  
    self.liblist   = [['libfftw3_mpi.a','libfftw3.a']]
    self.complex   = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps = [self.mpi]
    return
          
  def Install(self):
    import os

    args = ['--prefix='+self.installDir]
    args.append('--libdir='+os.path.join(self.installDir,self.libdir))
    self.framework.pushLanguage('C')
    ccompiler=self.framework.getCompiler()
    args.append('CC="'+self.framework.getCompiler()+'"')
    args.append('MPICC="'+self.framework.getCompiler()+'"')
    args.append('CFLAGS="'+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.framework.pushLanguage('Cxx')
      args.append('CXX="'+self.framework.getCompiler()+'"')
      args.append('CXXFLAGS="'+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
     # else error?
    if hasattr(self.compilers, 'FC'):
      self.framework.pushLanguage('FC')
      args.append('F77="'+self.framework.getCompiler()+'"')
      args.append('FFLAGS="'+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
     #else error?

    # MPI args need fixing
    args.append('--enable-mpi')
    if self.mpi.lib:
      args.append('LIBS="'+self.libraries.toStringNoDupes(self.mpi.lib)+'"')

    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'fftw'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('fftw'):
      try:
        self.logPrintBox('Configuring FFTW; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on FFTW: '+str(e))
      try:
        self.logPrintBox('Compiling FFTW; this may take several minutes')
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make && make install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on FFTW: '+str(e))
      self.postInstall(output1+err1+output2+err2,'fftw')
    return self.installDir
