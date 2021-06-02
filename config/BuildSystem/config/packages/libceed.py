import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit              = 'c5fbda681b280c9c826da7afffcb465658d8a54c' # main on May 31, 2021
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
    self.make            = framework.require('config.packages.make',self)
    return

  def Install(self):
    import os
    # TODO: maybe add support for various backends, CUDA, libXSMM, OCCA, MAGMA?
    with self.Language('C'):
      cc = self.getCompiler()
    try:
      self.logPrintBox('Compiling libceed; this may take several minutes')
      output,err,ret  = config.package.Package.executeShellCommand(self.make.make_jnp_list + ['CC='+cc, 'prefix='+self.installDir, '-B'], cwd=self.packageDir, timeout=250, log=self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running make on libceed: '+str(e))
    try:
      self.logPrintBox('Installing libceed; this may take several minutes')
      output,err,ret  = config.package.Package.executeShellCommand(self.make.make_sudo_list + ['install', 'prefix='+self.installDir], cwd=self.packageDir, timeout=250, log=self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running install on libceed: '+str(e))
    return self.installDir
