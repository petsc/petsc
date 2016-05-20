from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download        = ['https://sourceforge.net/projects/boost/files/boost/1.61.0/boost_1_61_0.tar.gz']
    self.includes        = ['boost/multi_index_container.hpp']
    self.liblist         = []
    self.cxx             = 1
    self.downloadonWindows = 1
    return

  def Install(self):
    import shutil
    import os

    conffile = os.path.join(self.packageDir,self.package+'.petscconf')
    fd = file(conffile, 'w')
    fd.write(self.installDir)
    fd.close()
    if not self.installNeeded(conffile): return self.installDir

    if not self.checkCompile('#include <bzlib.h>', ''):
      raise RuntimeError('Boost requires bzlib.h. Please install it in default compiler search location.')

    self.log.write('boostDir = '+self.packageDir+' installDir '+self.installDir+'\n')
    self.logPrintBox('Building and installing boost, this may take many minutes')
    self.installDirProvider.printSudoPasswordMessage()
    try:
      print 
      output,err,ret  = config.base.Configure.executeShellCommand('cd '+self.packageDir+'; ./bootstrap.sh --prefix='+self.installDir+'; ./b2 -j'+str(self.make.make_np)+';'+self.installSudo+'./b2 install', timeout=6000, log = self.log)
    except RuntimeError, e:
      raise RuntimeError('Error building/install Boost files from '+os.path.join(self.packageDir, 'Boost')+' to '+self.packageDir)
    self.postInstall(output+err,conffile)
    return self.installDir
