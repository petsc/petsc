from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download        = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/boost_minimal_1_42_0.tar.gz']
    self.includes        = ['boost/multi_index_container.hpp']
    self.cxx             = 1
    self.archIndependent = 1
    self.worksonWindows  = 1
    self.downloadonWindows = 1
    return

  def Install(self):
    import shutil
    import os
    self.framework.log.write('boostDir = '+self.packageDir+' installDir '+self.installDir+'\n')
    srcdir = os.path.join(self.packageDir,'boost')
    destdir = os.path.join(self.installDir,'include','boost')
    try:
      if os.path.isdir(destdir): shutil.rmtree(destdir)
      shutil.copytree(srcdir,destdir)
    except RuntimeError,e:
      raise RuntimeError('Error installing Boost include files: '+str(e))
    return self.installDir

  def getSearchDirectories(self):
    yield ''
    return
