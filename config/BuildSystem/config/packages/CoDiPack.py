import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit     = 'v1.9.1' # 2020-01-13
    self.download      = ['git://https://github.com/SciCompKL/CoDiPack.git']
    self.includes      = ['codi/adjointInterface.hpp']
    self.liblist       = []
    self.cxx           = 1
    self.minCxxVersion = 'c++11'
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    return

  def Install(self):
    import os

    self.logPrintBox('Copying CoDiPack include files to install location')
    self.installDirProvider.printSudoPasswordMessage()
    includedir = os.path.join(self.installDir, 'include')
    output2,err2,ret2  = config.package.Package.executeShellCommandSeq([
      self.withSudo('mkdir', '-p', includedir),
      self.withSudo('cp', '-rf', os.path.join('include', 'codi'), os.path.join('include', 'codi.hpp'), includedir),
      ], cwd=self.packageDir, timeout=250, log = self.log)
    return self.installDir
