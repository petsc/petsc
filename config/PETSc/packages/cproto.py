import PETSc.package
    
class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download         = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/cproto-4.6.tar.gz']
    self.complex          = 1
    self.double           = 0;
    self.requires32bitint = 0;
    
  def Install(self):
    import os
    if self.framework.argDB['with-batch']:
       args = ['--prefix='+self.installDir]
    else:
       args = ['--prefix='+self.installDir, '--with-cc='+'"'+self.setCompilers.CC+'"']          
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'cproto.args'), 'w')
    fd.write(args)
    fd.close()
    if self.installNeeded('cproto.args'):
      try:
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+';./configure '+args, timeout=900, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running configure on cproto: '+str(e))
      try:
        output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cd '+self.packageDir+';make; make install; make clean', timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make; make install on cproto: '+str(e))
      output,err,ret  = PETSc.package.NewPackage.executeShellCommand('cp -f '+os.path.join(self.packageDir,'cproto.args')+' '+self.confDir+'/cproto', timeout=5, log = self.framework.log)
      self.framework.actions.addArgument('CPROTO', 'Install', 'Installed cproto into '+self.installDir)
    self.binDir = os.path.join(self.installDir, 'bin')
    self.cproto = os.path.join(self.binDir, 'cproto')
    self.addMakeMacro('CPROTO',self.cproto)
    return self.installDir

  def alternateConfigureLibrary(self):
    self.checkDownload(1)
    
  def configure(self):
    '''Determine whether the cproto exist or not'''
    if self.framework.argDB.has_key('download-cproto') and self.framework.argDB['download-cproto']:

      self.getExecutable('cproto', getFullPath = 1)
      
      if hasattr(self, 'cproto'):
        self.addMakeMacro('CPROTO ', self.cproto)
        self.framework.logPrint('Found cproto, will not install cproto')
      else:
        self.framework.logPrint('Installing cproto')
        PETSc.package.NewPackage.configure(self)
    return
