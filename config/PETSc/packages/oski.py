import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download     = ['http://sourceforge.net/projects/oski/files/oski/1.0.1h/oski-1.0.1h.tar.gz/download']
    self.functions = ['oski_MatMult_Tid']
    self.includes  = ['oski/oski.h']
    self.liblist   = [['oski/liboski_Tid.a','oski/liboski.a']]
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.mpi,self.blasLapack]  
    return

  def Install(self):
    import os

    self.framework.pushLanguage('C')
    args = ['--prefix='+self.installDir]
    args.append('--libdir='+os.path.join(self.installDir,self.libdir))
    args.append('CC="'+self.framework.getCompiler()+'"')
    args.append('CFLAGS="'+self.framework.getCompilerFlags()+'"')
    self.framework.popLanguage()
    if hasattr(self.compilers, 'FC'):
      self.framework.pushLanguage('FC')
      args.append('F77="'+self.framework.getCompiler()+' '+self.framework.getCompilerFlags().replace('-Mfree','')+'"')
      self.framework.popLanguage()

    # crashes on Mac without this next lin
    args.append('--with-blas=no')

    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'oski'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('oski'):
      try:
        self.logPrintBox('Configuring oski; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on OSKI: '+str(e))
      try:
        self.logPrintBox('Compiling oski; this may take several minutes')
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('make && make benchmarks && make install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on OSKI: '+str(e))
      try:
        output3,err3,ret3  = PETSc.package.NewPackage.executeShellCommand(self.setCompilers.RANLIB+' '+os.path.join(self.installDir,'lib')+'/lib*.a', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running ranlib on OSKI libraries: '+str(e))
      self.postInstall(output1+err1+output2+err2+output3+err3,'oski')
    return self.installDir

