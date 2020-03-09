import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version           = '1.4.4'
    self.download          = ['https://github.com/facebook/zstd/archive/v'+self.version+'.tar.gz']
    self.functions         = ['ZSTD_compress']
    self.includes          = ['zstd.h']
    self.liblist           = [['libzstd.a']]
    self.downloaddirnames  = ['zstd']
    return

  def setupHelp(self, help):
    config.package.Package.setupHelp(self,help)
    import nargs
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers',self)
    self.make         = framework.require('config.packages.make',self)
    return

  def Install(self):
    import os
    with self.Language('C'):
      cc = self.setCompilers.getCompiler()
      cflags = self.setCompilers.getCompilerFlags()
    try:
      self.logPrintBox('Installing zstd; this may take several minutes')
      output,err,ret  = config.package.Package.executeShellCommand(self.make.make_jnp_list + ['CC='+cc, 'CFLAGS='+cflags, 'PREFIX='+self.installDir, 'install'], cwd=self.packageDir, timeout=250, log=self.log)
    except RuntimeError as e:
      raise RuntimeError('Error running make on zstd: '+str(e))
    return self.installDir
