import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit     = 'master'
    self.download      = ['git://https://github.com/SciCompKL/CoDiPack.git']
    self.includes      = ['adjointInterface.hpp']
    self.liblist       = []
    self.cxx           = 1
    self.requirescxx11 = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    return

  def Install(self):
    import os, glob

    self.logPrintBox('Copying CoDiPack include files to install location')
    self.installDirProvider.printSudoPasswordMessage()
    output2,err2,ret2  = config.package.Package.executeShellCommand([self.installSudo+'cp', '-rf'] + glob.glob('include/*') + [os.path.join(self.installDir,'include')], cwd=self.packageDir, timeout=250, log = self.log)
    return self.installDir
