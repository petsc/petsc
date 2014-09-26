from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.includes        = ['cusp/version.h']
    self.includedir      = ['','include']
    self.forceLanguage   = 'CUDA'
    self.cxx             = 0
    self.CUSPVersion     = '0400' # Minimal cusp version is 0.4 
    self.CUSPVersionStr  = str(int(self.CUSPVersion)/1000) + '.' + str(int(self.CUSPVersion)%100)
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.cuda = framework.require('config.packages.cuda', self)
    self.deps   = [self.cuda]
    return

  def getSearchDirectories(self):
    import os
    yield ''
    yield os.path.join('/usr','local','cuda')
    yield os.path.join('/usr','local','cuda','cusp')
    return

  def checkCUSPVersion(self):
    if 'known-cusp-version' in self.argDB:
      if self.argDB['known-cusp-version'] < self.CUSPVersion:
        raise RuntimeError('CUSP version error '+self.argDB['known-cusp-version']+' < '+self.CUSPVersion+': PETSC currently requires CUSP version '+self.CUSPVersionStr+' or higher')
    elif not self.argDB['with-batch']:
      self.pushLanguage('CUDA')
      oldFlags = self.compilers.CUDAPPFLAGS
      self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.include)
      if not self.checkRun('#include <cusp/version.h>\n#include <stdio.h>', 'if (CUSP_VERSION < ' + self.CUSPVersion +') {printf("Invalid version %d\\n", CUSP_VERSION); return 1;}'):
        raise RuntimeError('CUSP version error: PETSC currently requires CUSP version '+self.CUSPVersionStr+' or higher.')
      self.compilers.CUDAPPFLAGS = oldFlags
      self.popLanguage()
    else:
      raise RuntimeError('Batch configure does not work with CUDA\nOverride all CUDA configuration with options, such as --known-cusp-version')
    return

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does a additional tests needed by CUSP'''
    config.package.Package.configureLibrary(self)
    self.checkCUSPVersion()
    return

