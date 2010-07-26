from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download        = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/cusp_1_2.tar.gz']
    self.includes        = ['cusp/version.h']
    self.includedir      = ''
    self.cxx             = 0
    self.archIndependent = 1
    self.worksonWindows  = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.thrust = framework.require('config.packages.thrust', self)
    self.deps   = [self.thrust]
    return

  def Install(self):
    import shutil
    import os
    self.framework.log.write('cuspDir = '+self.packageDir+' installDir '+self.installDir+'\n')
    srcdir = os.path.join(self.packageDir, 'cusp')
    destdir = os.path.join(self.installDir, 'include', 'cusp')
    try:
      if os.path.isdir(destdir): shutil.rmtree(destdir)
      shutil.copytree(srcdir,destdir)
    except RuntimeError,e:
      raise RuntimeError('Error installing Cusp include files: '+str(e))
    return self.installDir

  def getSearchDirectories(self):
    yield ''
    return
