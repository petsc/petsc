import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download          = ['http://downloads.sourceforge.net/project/viennacl/1.7.x/ViennaCL-1.7.0.tar.gz',
                              'http://ftp.mcs.anl.gov/pub/petsc/externalpackages/ViennaCL-1.7.0.tar.gz' ]
    self.downloaddirname   = str('ViennaCL-1.7.0')
    self.includes          = ['viennacl/forwards.h']
    self.cxx               = 1
    self.downloadonWindows = 1
    self.complex           = 0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.opencl  = framework.require('config.packages.opencl',self)
    self.deps = [self.opencl]
    return

  def Install(self):
    import shutil
    import os
    self.log.write('ViennaCLDir = '+self.packageDir+' installDir '+self.installDir+'\n')
    #includeDir = self.packageDir
    srcdir     = os.path.join(self.packageDir, 'viennacl')
    destdir    = os.path.join(self.installDir, 'include', 'viennacl')
    if self.installSudo:
      self.installDirProvider.printSudoPasswordMessage()
      try:
        output,err,ret  = config.package.Package.executeShellCommand(self.installSudo+'mkdir -p '+destdir+' && '+self.installSudo+'rm -rf '+destdir+'  && '+self.installSudo+'cp -rf '+srcdir+' '+destdir, timeout=6000, log = self.log)
      except RuntimeError, e:
        raise RuntimeError('Error copying ViennaCL include files from '+os.path.join(self.packageDir, 'ViennaCL')+' to '+packageDir)
    else:
      try:
        if os.path.isdir(destdir): shutil.rmtree(destdir)
        shutil.copytree(srcdir,destdir)
      except RuntimeError,e:
        raise RuntimeError('Error installing ViennaCL include files: '+str(e))
    return self.installDir

