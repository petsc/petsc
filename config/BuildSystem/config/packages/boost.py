from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download          = ['http://downloads.sourceforge.net/project/boost/boost/1.61.0/boost_1_61_0.tar.gz',
                              'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/boost_1_61_0.tar.gz']
    self.includes          = ['boost/multi_index_container.hpp']
    self.liblist           = []
    self.cxx               = 1
    self.downloadonWindows = 1
    self.useddirectly      = 0
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('BOOST', '-boost-headers-only=<bool>', nargs.ArgBool(None, 0, 'When true, do not build boost libraries, only install headers'))

  def Install(self):
    import shutil
    import os

    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = open(conffile, 'w')
    fd.write(self.installDir)
    fd.close()
    if not self.installNeeded(conffile): return self.installDir

    if self.framework.argDB['boost-headers-only']:
       boostIncludeDir = os.path.join(os.path.join(self.installDir, self.includedir), 'boost')
       self.logPrintBox('Configure option --boost-headers-only is ENABLED ... boost libraries will not be built')
       self.logPrintBox('Installing boost headers, this should not take long')
       try:
         if os.path.lexists(boostIncludeDir): os.remove(boostIncludeDir)
         output,err,ret  = config.base.Configure.executeShellCommand('cd '+self.packageDir+';' + 'ln -s $PWD/boost/ ' + boostIncludeDir, timeout=6000, log = self.log)
       except RuntimeError as e:
         raise RuntimeError('Error linking '+self.packageDir+' to '+ boostIncludeDir)
       return self.installDir
    else:
       if not self.checkCompile('#include <bzlib.h>', ''):
         raise RuntimeError('Boost requires bzlib.h. Please install it in default compiler search location.')

       self.log.write('boostDir = '+self.packageDir+' installDir '+self.installDir+'\n')
       self.logPrintBox('Building and installing boost, this may take many minutes')
       self.installDirProvider.printSudoPasswordMessage()
       try:
         output,err,ret  = config.base.Configure.executeShellCommand('cd '+self.packageDir+'; ./bootstrap.sh --prefix='+self.installDir+'; ./b2 -j'+str(self.make.make_np)+';'+self.installSudo+'./b2 install', timeout=6000, log = self.log)
       except RuntimeError as e:
         raise RuntimeError('Error building/install Boost files from '+os.path.join(self.packageDir, 'Boost')+' to '+self.packageDir)
       self.postInstall(output+err,conffile)
    return self.installDir
