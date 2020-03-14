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
      output1,err1,ret1  = config.package.Package.executeShellCommand(self.make.make_jnp_list + ['clean', 'all'], cwd=os.path.join(self.packageDir,'src'), timeout=60, log = self.log)
      libdir = os.path.join(self.installDir, 'lib')
      includedir = os.path.join(self.installDir, 'lib')
      output2,err2,ret2  = config.package.Package.executeShellCommandSeq([
        self.withSudo('mkdir', '-p', libdir, includedir),
        self.withSudo('cp', '-f', os.path.join('src', 'libadblaslapack.a'), libdir),
        self.withSudo('cp', '-f', os.path.join('include', 'adblaslapack.hpp'), includedir),
        ], cwd=self.packageDir, timeout=60, log = self.log)
      self.postInstall(output1+err1+output2+err2,'Makefile.inc')
    return self.installDir
