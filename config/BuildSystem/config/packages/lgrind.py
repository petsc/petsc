import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download     = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/lgrind-dev.tar.gz']
    self.linkedbypetsc     = 0
    self.useddirectly      = 0
    self.executablename    = 'lgrind'
    #
    #  lgrind is currently not used by PETSc
    #
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.petscclone     = framework.require('PETSc.options.petscclone',self.setCompilers)
    return

  def Install(self):
    import os
    try:
      self.framework.pushLanguage('C')
      output,err,ret = config.package.Package.executeShellCommand('cd '+os.path.join(self.packageDir,'source')+' && make clean && make CC=\''+self.framework.getCompiler()+'\'',timeout=2500,log = self.log)
      self.framework.popLanguage()
    except RuntimeError as e:
      self.framework.popLanguage()
      if self.argDB['with-batch']:
        self.logPrintBox('Batch build that could not generate lgrind, you may not be able to build all documentation')
        return
      raise RuntimeError('Error running make on lgrind: '+str(e))
    output,err,ret  = config.package.Package.executeShellCommand('cp -f '+os.path.join(self.packageDir,'source','lgrind')+' '+os.path.join(self.confDir,'bin'), timeout=60, log = self.log)
    output,err,ret  = config.package.Package.executeShellCommand('cp -f '+os.path.join(self.packageDir,'lgrind.sty')+' '+os.path.join(self.confDir,'share'), timeout=60, log = self.log)
    output,err,ret  = config.package.Package.executeShellCommand('cp -f '+os.path.join(self.packageDir,'lgrindef')+' '+os.path.join(self.confDir,'share'), timeout=60, log = self.log)
    return self.confDir

  def configure(self):
    import os
    '''Determine location of lgrind, download if requested'''
    if self.petscclone.isClone:
      self.getExecutable('lgrind', getFullPath = 1)

      if hasattr(self, 'lgrind') and not self.argDB['download-lgrind']:
        self.logPrint('Found lgrind, will not install lgrind')
      elif self.argDB['download-lgrind']:
        config.package.Package.configure(self)
        self.getExecutable('lgrind',    path=os.path.join(self.installDir,'bin'), getFullPath = 1)
    else:
      self.logPrint("Not a clone of PETSc, don't need Lgrind\n")
    return
