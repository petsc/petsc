import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download     = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/hypre-2.8.0b.tar.gz']
    self.functions = ['HYPRE_IJMatrixCreate']
    self.includes  = ['HYPRE.h']
    self.liblist   = [['libHYPRE.a']]
    self.license   = 'https://computation.llnl.gov/casc/linear_solvers/sls_hypre.html'
    # Per hypre users guide section 7.5 - install manually on windows for MS compilers.
    self.worksonWindows   = 1
    self.downloadonWindows= 0
    self.requires32bitint = 0

    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]  
    return

  def generateLibList(self,dir):
    '''Normally the one in package.py is used, but hypre requires the extra C++ library'''
    alllibs = PETSc.package.NewPackage.generateLibList(self,dir)
    import config.setCompilers
    if self.languages.clanguage == 'C':
      alllibs[0].extend(self.compilers.cxxlibs)
    return alllibs
        
  def Install(self):
    import os

    self.framework.pushLanguage('C')
    args = ['--prefix='+self.installDir]
    args.append('--libdir='+os.path.join(self.installDir,self.libdir))
    args.append('CC="'+self.framework.getCompiler()+'"')
    args.append('CFLAGS="'+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
    if hasattr(self.compilers, 'CXX'):
      self.framework.pushLanguage('Cxx')
      args.append('CXX="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'"')
      self.framework.popLanguage()
    else:
      raise RuntimeError('Error: Hypre requires C++ compiler. None specified')
    if hasattr(self.compilers, 'FC'):
      self.framework.pushLanguage('FC')
      args.append('F77="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags().replace('-Mfree','')+'"')
      self.framework.popLanguage()
    else:
      raise RuntimeError('Error: Hypre requires Fortran compiler. None specified (was your MPI built with Fortran support?')
    if self.mpi.include:
      # just use the first dir - and assume the subsequent one isn't necessary [relavant only on AIX?]
      args.append('--with-MPI-include="'+self.mpi.include[0]+'"')
    libdirs = []
    for l in self.mpi.lib:
      ll = os.path.dirname(l)
      libdirs.append(ll)
    libdirs = ' '.join(libdirs)
    args.append('--with-MPI-lib-dirs="'+libdirs+'"')
    libs = []
    for l in self.mpi.lib:
      ll = os.path.basename(l)
      libs.append(ll[3:-2])
    libs = ' '.join(libs)
    args.append('--with-MPI-libs="'+libs+'"')

    # tell hypre configure not to look for blas/lapack [and not use hypre-internal blas]
    args.append('--with-blas-libs=')
    args.append('--with-blas-lib-dir=')
    args.append('--with-lapack-libs=')
    args.append('--with-lapack-lib-dir=')
    args.append('--with-blas=yes')
    args.append('--with-lapack=yes')

    # explicitly tell hypre BLAS/LAPACK mangling since it may not match Fortran mangling
    if self.blasLapack.mangling == 'underscore':
      mang = 'one-underscore'
    elif self.blasLapack.mangling == 'caps':
      mang = 'caps-no-underscores'
    else:
      mang = 'no-underscores'
    args.append('--with-fmangle-blas='+mang)
    args.append('--with-fmangle-lapack='+mang)

    args.append('--without-babel')
    args.append('--without-mli')
    args.append('--without-fei')
    args.append('--without-superlu')
    if self.libraryOptions.integerSize == 64:
      args.append('--enable-bigint')
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'hypre'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('hypre'):
      try:
        self.logPrintBox('Configuring hypre; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+' && touch config/Makefile.config && make distclean && ./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on HYPRE: '+str(e))
      try:
        self.logPrintBox('Compiling hypre; this may take several minutes')
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+' && make install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on HYPRE: '+str(e))
      try:
        output3,err3,ret3  = PETSc.package.NewPackage.executeShellCommand(self.setCompilers.RANLIB+' '+os.path.join(self.installDir,'lib')+'/lib*.a', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running ranlib on HYPRE libraries: '+str(e))
      self.postInstall(output1+err1+output2+err2+output3+err3,'hypre')
    return self.installDir

  def consistencyChecks(self):
    PETSc.package.NewPackage.consistencyChecks(self)
    if self.framework.argDB['with-'+self.package]:
      if not self.blasLapack.checkForRoutine('dgels'):
        raise RuntimeError('hypre requires the LAPACK routine dgels(), the current Lapack libraries '+str(self.blasLapack.lib)+' does not have it')
      self.framework.log.write('Found dgels() in Lapack library as needed by hypre\n')
    return
