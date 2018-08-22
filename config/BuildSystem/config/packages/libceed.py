import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit              = 'master'  #master+
    self.download               = ['git://https://github.com/CEED/libceed.git']
    self.functions              = ['CeedRegister']
    self.includes               = ['ceed.h']
    self.liblist                = [['libceed.a']]
    self.requires32bitint       = 1;   # TODO: fix for 64 bit integers
    return

  def setupHelp(self, help):
    config.package.Package.setupHelp(self,help)
    import nargs
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.setCompilers    = framework.require('config.setCompilers',self)
    self.installdir      = framework.require('PETSc.options.installDir',self)
    return

  def Install(self):
    import os
    self.pushLanguage('C')
    # TODO: maybe add support for various backends, OCCA, MAGMA?
    cc = self.setCompilers.getCompiler()
    self.popLanguage()
    try:
      self.logPrintBox('Compiling libceed; this may take several minutes')
      output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && make clean && CC='+cc+' prefix='+self.installDir+' make ', timeout=250, log = self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running make on libceed: '+str(e))
    try:
      self.logPrintBox('Installing libceed; this may take several minutes')
      output,err,ret  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && '+self.installSudo+' prefix='+self.installDir+' make install', timeout=250, log = self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running install on libceed: '+str(e))
    return self.installDir


