import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.gitcommit = '631743a8d669f0259324622fd6aca29cfe58f659' #main july-02-2021
    self.download  = ['git://https://github.com/caidao22/cams.git']
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
      try:
        output1,err1,ret1 = config.package.Package.executeShellCommand(self.make.make_jnp_list + ['clean', 'lib'], cwd=self.packageDir, timeout=250, log=self.log)
      except RuntimeError as e:
        raise RuntimeError('Error running make on CAMS: '+str(e))
      try:
        output2,err2,ret2 = config.package.Package.executeShellCommand(self.make.make_jnp_list + ['install'], cwd=self.packageDir, timeout=250, log=self.log)
      except RuntimeError as e:
        raise RuntimeError('Error running install on CAMS: '+str(e))
      self.postInstall(output1+err1+output2+err2, 'make.inc')
    return self.installDir
