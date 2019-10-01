import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit       = '388077c' #master oct-01-2019
    self.gitcommitmaster = 'origin/master'
    self.download        = ['git://https://github.com/SciCompKL/CoDiPack.git']
    self.includes        = ['codi/adjointInterface.hpp']
    self.liblist         = []
    self.cxx             = 1
    self.requirescxx11   = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    return

  def Install(self):
    import os

    self.logPrintBox('Copying CoDiPack include files to install location')
    self.installDirProvider.printSudoPasswordMessage()
    output2,err2,ret2  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && '+self.installSudo+' cp -rf include/* '+os.path.join(self.installDir,'include'),timeout=250, log = self.log)
    return self.installDir
