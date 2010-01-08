import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download         = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/mpack-0.6.0.tar.gz']
    self.functions        = ['']
    self.includes         = ['']
    self.liblist          = [['']]
    self.needsMath        = 1
    self.complex          = 1
    self.cxx              = 0
    self.double           = 0      
    self.requires32bitint = 1;    
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.qd    = framework.require('config.packages.qd',self)
    self.deps  = [self.qd]
    return

  def Install(self):
    import os

    args = []
    self.framework.pushLanguage('Cxx')
    args.append('--prefix='+self.installDir)
    args.append('CXX="'+self.framework.getCompiler()+'"')
    args.append('CFLAGS="'+self.framework.getCompilerFlags()+'"')
    args.append('--with-qd-includedir='+self.qd.includeDir)
    args.append('--with-qd-libdir='+self.qd.libDir)    
    self.framework.popLanguage()
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'mpack'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('mpack'):
      try:
        self.logPrintBox('Configuring MPACK; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+';./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on MPACK: '+str(e))
      try:
        self.logPrintBox('Compiling MPACK; this may take several minutes')
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+'; make ; make install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on MPACK: '+str(e))
      self.postInstall(output1+err1+output2+err2,'mpack')
    return self.installDir

