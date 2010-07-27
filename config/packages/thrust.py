from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download        = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/thrust_0_1.tar.gz']
    self.includes        = ['thrust/version.h']
    self.includedir      = ''
    self.forceLanguage   = 'CUDA'
    self.cxx             = 0
    self.archIndependent = 1
    self.worksonWindows  = 1
    return

  def Install(self):
    import shutil
    import os
    self.framework.log.write('thrustDir = '+self.packageDir+' installDir '+self.installDir+'\n')
    srcdir = os.path.join(self.packageDir, 'thrust')
    destdir = os.path.join(self.installDir, 'include', 'thrust')
    try:
      if os.path.isdir(destdir): shutil.rmtree(destdir)
      shutil.copytree(srcdir,destdir)
    except RuntimeError,e:
      raise RuntimeError('Error installing Thrust include files: '+str(e))
    return self.installDir

  def getSearchDirectories(self):
    yield ''
    return
