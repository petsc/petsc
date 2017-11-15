import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
#    self.download     = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/expat-2.0.0.tar.gz']
    self.download     = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/expat-1.95.8.tar.gz']
    self.functions = ['XML_ExpatVersion']
    self.liblist   = [['libexpat.a']]
    self.includes  = ['expat.h']
    return

  def Install(self):
    import os

    self.framework.pushLanguage('C')
    flags = self.framework.getCompilerFlags()
    #  expat autoconf turns on GCC options if it thinks you are using a GNU compile :-(
    if config.setCompilers.Configure.isIntel(self.framework.getCompiler()):
      flags = flags + ' -gcc-sys'
    args = ['--prefix='+self.installDir, 'CC="'+self.framework.getCompiler()+' '+flags+'"']
    args.append('--libdir='+os.path.join(self.installDir,self.libdir))
    self.framework.popLanguage()
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'expat'), 'w')
    fd.write(args)
    fd.close()

    if self.installNeeded('expat'):
      try:
        self.logPrintBox('Configuring expat; this may take several minutes')
        output1,err1,ret1  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && ./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on expat: '+str(e))
      try:
        self.logPrintBox('Compiling expat; this may take several minutes')
        output2,err2,ret2  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+' && make && make install', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on expat: '+str(e))
      try:
        output3,err3,ret3  = PETSc.package.NewPackage.executeShellCommand(self.setCompilers.RANLIB+' '+os.path.join(self.installDir,'lib')+'/lib*.a', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running ranlib on expat libraries: '+str(e))
      self.postInstall(output1+err1+output2+err2+output3+err3,'expat')
    return self.installDir
