from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download        = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/thrust_100_201.tar.gz']
    self.includes        = ['thrust/version.h']
    self.includedir      = ''
    self.forceLanguage   = 'CUDA'
    self.cxx             = 0
    self.archIndependent = 1
    self.worksonWindows  = 1
    self.version         = '100201'
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
    import os
    yield [os.path.join('/usr','local','cuda')]
    yield ''
    return

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional tests needed by Thrust'''
    config.package.Package.configureLibrary(self)
    self.executeTest(self.checkVersion)
    return

  def checkVersion(self):
    self.pushLanguage('CUDA')
    if not self.checkRun('#include <thrust/version.h>\n', 'if (THRUST_VERSION != ' + self.version +') return 1'):
      raise RuntimeError('Thrust version error: PETSC currently requires Thrust version '+ str(int(self.version)/100000) + '.' + str(int(self.version) / 100 % 1000) + '.' + str(int(self.version) % 100) + ' when compiling with CUDA')
    self.popLanguage()
    return
