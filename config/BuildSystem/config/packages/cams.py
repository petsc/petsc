import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit = 'baa7d400003c906fe8ba72076190d4445f40c0f9' #main june-18-2021
    self.download  = ['git://https://github.com/caidao22/pkg-cams.git']
    self.functions = ['offline_cams_create']
    self.includes  = ['offline_schedule.h']
    self.liblist   = [['libcams.a']]
    self.hastests  = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    return

  def Install(self):
    import os

    g = open(os.path.join(self.packageDir,'make.inc'),'w')
    g.write('CP               = '+self.programs.cp+'\n')
    g.write('RM               = '+self.programs.RM+'\n')
    g.write('MKDIR            = '+self.programs.mkdir+'\n')

    g.write('AR               = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS          = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('AR_LIB_SUFFIX    = '+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('RANLIB           = '+self.setCompilers.RANLIB+'\n')

    g.write('PREFIX           = '+self.installDir+'\n')

    self.pushLanguage('Cxx')
    g.write('CXX               = '+self.getCompiler()+'\n')
    g.write('CXXFLAGS          = '+self.updatePackageCxxFlags(self.getCompilerFlags())+'\n')
    self.popLanguage()
    self.pushLanguage('C')
    g.write('CC                = '+self.getCompiler()+'\n')
    g.write('CFLAGS           = '+self.updatePackageCFlags(self.getCompilerFlags())+'\n')
    self.popLanguage()
    g.close()

    if self.installNeeded('make.inc'):
      self.logPrintBox('Configuring, compiling and installing cams; this may take several seconds')
      self.installDirProvider.printSudoPasswordMessage()
      output1,err1,ret1  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && make clean && make lib',timeout=500, log = self.log)
      output2,err2,ret2  = config.package.Package.executeShellCommand('cd '+self.packageDir+' && '+self.installSudo+' make install ',timeout=250, log = self.log)
      self.postInstall(output1+err1+output2+err2,'make.inc')
    return self.installDir
