import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download  = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/sundials-2.5.0.tar.gz']
    self.functions = ['CVSpgmr']
    self.includes  = ['sundials/sundials_nvector.h'] 
    self.liblist   = [['libsundials_cvode.a','libsundials_nvecserial.a','libsundials_nvecparallel.a']] #currently only support CVODE
    self.license   = 'http://www.llnl.gov/CASC/sundials/download/download.html'
    self.needsMath = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps = [self.mpi,self.blasLapack]
    return
          
  def Install(self):
    import os
    self.framework.pushLanguage('C')
    ccompiler=self.framework.getCompiler()
    args = ['--prefix='+self.installDir]
    args.append('--libdir='+os.path.join(self.installDir,self.libdir))
    args.append('CC="'+self.framework.getCompiler()+'"')
    args.append('CFLAGS="'+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.framework.pushLanguage('Cxx')
      args.append('CXX="'+self.framework.getCompiler()+'"')
      args.append('CXXFLAGS="'+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    
    # use --with-mpi-root if we know it works
    if self.mpi.directory and (os.path.realpath(ccompiler)).find(os.path.realpath(self.mpi.directory)) >=0:
      self.framework.log.write('Sundials configure: using --with-mpi-root='+self.mpi.directory+'\n')
      args.append('--with-mpi-root="'+self.mpi.directory+'"')
    # else provide everything!
    else:
      #print a message if the previous check failed
      if self.mpi.directory:
        self.framework.log.write('Sundials configure: --with-mpi-dir specified - but could not use it\n')
        self.framework.log.write(str(os.path.realpath(ccompiler))+' '+str(os.path.realpath(self.mpi.directory))+'\n')
        
      args.append('--without-mpicc')  
      if self.mpi.include:
        args.append('--with-mpi-incdir="'+self.mpi.include[0]+'"')
      else: 
        args.append('--with-mpi-incdir="/usr/include"')  # dummy case

      if self.mpi.lib:
        args.append('--with-mpi-libdir="'+os.path.dirname(self.mpi.lib[0])+'"')
        libs = []
        for l in self.mpi.lib:
          ll = os.path.basename(l)
          if ll.endswith('.a'): libs.append(ll[3:-2])
          elif ll.endswith('.so'): libs.append(ll[3:-3])
          elif ll.endswith('.dylib'): libs.append(ll[3:-6])
          libs.append(ll[3:-2])
        libs = '-l' + ' -l'.join(libs)
        args.append('--with-mpi-libs="'+libs+'"')
      else:
        args.append('--with-mpi-libdir="/usr/lib"')  # dummy case
        args.append('--with-mpi-libs="-lc"')
   
    args.append('--without-mpif77')
    args.append('--disable-examples')
    args.append('--disable-cvodes')
    args.append('--disable-ida')
    args.append('--disable-kinsol')
    args.append('--disable-f77')
    args.append('--disable-libtool-lock')
    
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'sundials'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('sundials'):
      try:
        self.logPrintBox('Configuring sundials; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on SUNDIALS: '+str(e))
      try:
        self.logPrintBox('Compiling sundials; this may take several minutes')
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make && make install && make clean', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on SUNDIALS: '+str(e))
      self.postInstall(output1+err1+output2+err2,'sundials')
    return self.installDir
