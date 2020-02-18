import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit  = 'master'
    self.download   = ['git://https://xgitlab.cels.anl.gov/schanen/adblaslapack.git']
    self.functions  = []
    self.includes   = []
    self.liblist    = [['libadblaslapack.a']]
    self.cxx        = 1
    self.precisions = ['double']
    self.complex    = 0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.CoDiPack = framework.require('config.packages.CoDiPack',self)
    self.deps     = [self.CoDiPack]
    return

  def Install(self):
    import os

    self.framework.pushLanguage('Cxx')
    g = open(os.path.join(self.packageDir,'Makefile.inc'),'w')
    g.write('CODI_DIR         = '+self.CoDiPack.include[0]+'\n')
    g.write('AROPT            = rcs\n')
    g.write('AR               = '+self.setCompilers.AR+'\n')
    g.write('CXX              = '+self.framework.getCompiler()+'\n')
    g.write('CFLAGS           = -I$(CODI_DIR) -I../include '+self.removeWarningFlags(self.framework.getCompilerFlags())+'\n')
    g.close()
    self.framework.popLanguage()

    if self.installNeeded('Makefile.inc'):
      self.logPrintBox('Configuring, compiling and installing adblaslapack; this may take several seconds')
      self.installDirProvider.printSudoPasswordMessage()
      output1,err1,ret1  = config.package.Package.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+' && make clean all ',timeout=60, log = self.log)
      output2,err2,ret2  = config.package.Package.executeShellCommand('cd '+os.path.join(self.packageDir,'src')+' && '+self.installSudo+' cp -f libadblaslapack.a '+os.path.join(self.installDir,'lib'),timeout=60, log = self.log)
      output2,err2,ret2  = config.package.Package.executeShellCommand('cd '+os.path.join(self.packageDir,'include')+' && '+self.installSudo+' cp -f adblaslapack.hpp '+os.path.join(self.installDir,'include'),timeout=60, log = self.log)
      self.postInstall(output1+err1+output2+err2,'Makefile.inc')
    return self.installDir
