import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.download        = ['http://downloads.sourceforge.net/project/viennacl/1.4.x/ViennaCL-1.4.1.tar.gz']
    self.downloadfilename = str('ViennaCL-1.4.1')
    self.includes        = ['viennacl/forwards.h']
    self.cxx             = 1
    self.worksonWindows  = 1
    self.downloadonWindows = 1
    self.complex          = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.opencl  = framework.require('PETSc.packages.opencl',self)
    self.deps = [self.opencl]
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

