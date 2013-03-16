from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download        = ['http://downloads.sourceforge.net/project/viennacl/1.4.x/ViennaCL-1.4.1.tar.gz']
    self.downloadfilename = str('ViennaCL-1.4.1')
    self.includes        = ['viennacl/forwards.h']
    self.includedir      = ['.']
    self.cxx             = 1
    self.archIndependent = 1
    self.worksonWindows  = 1
    self.downloadonWindows = 1
    return

  def Install(self):
    import shutil
    import os
    self.framework.log.write('ViennaCLDir = '+self.packageDir+' installDir '+self.installDir+'\n')
    #includeDir = self.packageDir
    srcdir     = os.path.join(self.packageDir, 'viennacl')
    destdir    = os.path.join(self.installDir, 'include', 'viennacl')
    try:
      if os.path.isdir(destdir): shutil.rmtree(destdir)
      shutil.copytree(srcdir,destdir)
    except RuntimeError,e:
      raise RuntimeError('Error installing ViennaCL include files: '+str(e))
    return self.installDir

  def getSearchDirectories(self):
    yield ''
    return
