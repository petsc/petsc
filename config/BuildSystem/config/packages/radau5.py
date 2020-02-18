import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit         = 'master'
    self.download          = ['git://https://bitbucket.org/petsc/pkg-radau5.git','https://bitbucket.org/petsc/pkg-radau5/get/'+self.gitcommit+'.tar.gz']
    self.liblist           = [['libradau5.a']]
    self.libdir            = ''
    self.precisions        = ['double']
    self.requires32bitint  = 1;
    self.complex           = 0;
    self.fc                = 1;
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.petscclone     = framework.require('PETSc.options.petscclone',self.setCompilers)
    return

  def Install(self):
    import os
    try:
      self.framework.pushLanguage('FC')
      output,err,ret = config.package.Package.executeShellCommand('cd '+self.packageDir+' && make AR=ar FC=\''+self.framework.getCompiler()+' '+self.framework.getCompilerFlags()+'\'',timeout=2500,log = self.log)
      self.framework.popLanguage()
    except RuntimeError as e:
      raise RuntimeError('Error running make on radau5: '+str(e))
    output,err,ret  = config.package.Package.executeShellCommand('cp -f '+os.path.join(self.packageDir,'libradau5.a')+' '+os.path.join(self.confDir,'lib'), timeout=60, log = self.log)
    return os.path.join(self.confDir,'lib')

