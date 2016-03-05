from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download        = ['http://downloads.sourceforge.net/project/boost/boost/1.60.0/boost_1_60_0.tar.gz']
    self.includes        = ['boost/multi_index_container.hpp']
    self.liblist         = []
    self.cxx             = 1
    self.downloadonWindows = 1
    return

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if self.argDB.get('download-'+self.downloadname.lower()) and self.setCompilers.isDarwin(self.log):
      raise RuntimeError('--download-boost does not produce correct shared libraries on Apple. Suggest:\n   brew install boost\nthen run ./configure with --with-boost-dir=/usr/local\n')

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
