from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version         = '102' #Version 0.1.2
    self.versionStr       = str(int(self.version)/100000) + '.' + str(int(self.version)/100%1000) + '.' + str(int(self.version)%100)
    self.download        = ['http://cusp-library.googlecode.com/files/cusp-v'+self.versionStr+'.zip']
    self.includes        = ['cusp/version.h']
    self.includedir      = ''
    self.forceLanguage   = 'CUDA'
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
    srcdir = self.packageDir
    destdir = os.path.join(self.installDir, 'include', 'cusp')
    try:
      if os.path.isdir(destdir): shutil.rmtree(destdir)
      shutil.copytree(srcdir,destdir)
    except RuntimeError,e:
      raise RuntimeError('Error installing Cusp include files: '+str(e))
    self.includedir = 'include' # default and --download have different includedirs
    return self.installDir

  def getSearchDirectories(self):
    import os
    yield os.path.join('/usr','local','cuda')
    yield ''
    return

  def configurePC(self):
    self.pushLanguage('CUDA')
    if self.checkCompile('#include <cusp/version.h>\n#include <cusp/precond/smoothed_aggregation.h>\n', ''):
      self.addDefine('HAVE_CUSP_SMOOTHED_AGGREGATION','1')
    self.popLanguage()
    return

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional tests needed by CUSP'''
    if not self.thrust.found:
      raise RuntimeError('CUSP support requires the THRUST package\nRerun configure using --with-thrust-dir')
    config.package.Package.configureLibrary(self)
    self.executeTest(self.configurePC)
    self.executeTest(self.checkVersion)
    return

  def checkVersion(self):
    self.pushLanguage('CUDA')
    oldFlags = self.compilers.CUDAPPFLAGS
    self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.include)
    self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.thrust.include)
    if not self.checkRun('#include <cusp/version.h>\n#include <stdio.h>', 'if (CUSP_VERSION < ' + self.version +') {printf("Invalid version %d\\n", CUSP_VERSION); return 1;}'):
      raise RuntimeError('Cusp version error: PETSC currently requires Cusp version '+self.versionStr+' when compiling with CUDA')
    self.compilers.CUDAPPFLAGS = oldFlags
    self.popLanguage()
    return
