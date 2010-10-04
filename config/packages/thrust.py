from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version         = '100201' #Version 1.2.1
    self.versionStr       = str(int(self.version)/100000) + '.' + str(int(self.version)/100%1000) + '.' + str(int(self.version)%100)
    self.download        = ['http://thrust.googlecode.com/files/thrust-v'+self.versionStr+'.zip']
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
    srcdir = self.packageDir
    destdir = os.path.join(self.installDir, 'include', 'thrust')
    try:
      if os.path.isdir(destdir): shutil.rmtree(destdir)
      shutil.copytree(srcdir,destdir)
    except RuntimeError,e:
      raise RuntimeError('Error installing Thrust include files: '+str(e))
    self.includedir = 'include' # default and --download have different includedirs
    return self.installDir

  def getSearchDirectories(self):
    import os
    yield os.path.join('/usr','local','cuda')
    yield ''
    return

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional tests needed by Thrust'''
    config.package.Package.configureLibrary(self)
    self.executeTest(self.checkVersion)
    return

  def checkVersion(self):
    self.pushLanguage('CUDA')
    oldFlags = self.compilers.CUDAPPFLAGS
    self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.include)
    if not self.checkRun('#include <thrust/version.h>\n#include <stdio.h>', 'if (THRUST_VERSION < ' + self.version +') {printf("Invalid version %d\\n", THRUST_VERSION); return 1;}'):
      raise RuntimeError('Thrust version error: PETSC currently requires Thrust version '+self.versionStr+' when compiling with CUDA')
    self.compilers.CUDAPPFLAGS = oldFlags
    self.popLanguage()
    return
