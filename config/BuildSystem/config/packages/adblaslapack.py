import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit  = 'e55b6ad4234066617ef198cbf080f0d07d151823' #master jul-02-2018
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

    self.pushLanguage('Cxx')
    g = open(os.path.join(self.packageDir,'Makefile.inc'),'w')
    g.write('CODI_DIR         = '+self.CoDiPack.include[0]+'\n')
    g.write('AROPT            = rcs\n')
    g.write('AR               = '+self.setCompilers.AR+'\n')
    g.write('CXX              = '+self.getCompiler()+'\n')
    g.write('CFLAGS           = -I$(CODI_DIR) -I../include '+self.updatePackageCFlags(self.getCompilerFlags())+'\n')
    g.close()
    self.popLanguage()

    if self.installNeeded('Makefile.inc'):
      self.logPrintBox('Configuring, compiling and installing adblaslapack; this may take several seconds')
      output1,err1,ret1  = config.package.Package.executeShellCommand(self.make.make_jnp_list + ['clean', 'all'], cwd=os.path.join(self.packageDir,'src'), timeout=60, log = self.log)
      libdir = os.path.join(self.installDir, 'lib')
      includedir = os.path.join(self.installDir, 'lib')
      output2,err2,ret2  = config.package.Package.executeShellCommandSeq([
        ['mkdir', '-p', libdir, includedir],
        ['cp', '-f', os.path.join('src', 'libadblaslapack.a'), libdir],
        ['cp', '-f', os.path.join('include', 'adblaslapack.hpp'), includedir],
        ], cwd=self.packageDir, timeout=60, log = self.log)
      self.postInstall(output1+err1+output2+err2,'Makefile.inc')
    return self.installDir
