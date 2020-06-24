import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    # disable version check
    #self.version          = '1.8.0'
    #self.minversion       = '1.8.0'
    #self.versionname      = ???
    #self.gitcommit        = 'v'+self.version
    version               = '1.8.0'
    self.gitcommit        = 'v'+version
    self.download         = ['git://https://github.com/NVlabs/cub','https://github.com/NVlabs/cub/archive/'+self.gitcommit+'.tar.gz']
    self.functions        = []
    self.includes         = ['cub/cub.cuh']
    self.precisions       = ['single','double']
    self.cxx              = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    return

  def Install(self):
    import os

    with open(os.path.join(self.packageDir,'petsc.mk'),'w') as g:
      g.write('#empty\n')

    if not self.installNeeded('petsc.mk'):
      return self.installDir

    if self.framework.argDB['prefix'] and not 'package-prefix-hash' in self.argDB:
      PETSC_DIR  = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
      PETSC_ARCH = ''
      prefix     = os.path.abspath(os.path.expanduser(self.argDB['prefix']))
    else:
      PETSC_DIR  = self.petscdir.dir
      PETSC_ARCH = self.arch
      prefix     = os.path.join(self.petscdir.dir,self.arch)
    incDir = os.path.join(prefix,'include')
    if self.installSudo:
       newuser = self.installSudo+' -u $${SUDO_USER} '
    else:
       newuser = ''
    cpstr = newuser+' mkdir -p '+incDir+' && '+newuser+' cp -r '+os.path.join(self.packageDir,'cub')+' '+incDir
    try:
      self.logPrintBox('Copying CUB; this may take several seconds')
      output,err,ret = config.package.Package.executeShellCommand(cpstr,timeout=100,log=self.log)
    except RuntimeError as e:
      self.logPrint('Error executing "'+cpstr+'": '+str(e))
      raise RuntimeError('Error copying CUB')
    self.postInstall(output+err,'petsc.mk')
    return self.installDir
