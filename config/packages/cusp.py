from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
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

  def getSearchDirectories(self):
    import os
    yield ''
    yield os.path.join('/usr','local','cuda')
    yield os.path.join('/usr','local','cuda','cusp')
    return

  def configurePC(self):
    self.pushLanguage('CUDA')
    oldFlags = self.compilers.CUDAPPFLAGS
    self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.include)
    self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.thrust.include)
    if self.checkCompile('#include <cusp/version.h>\n#include <cusp/precond/smoothed_aggregation.h>\n', ''):
      self.addDefine('HAVE_CUSP_SMOOTHED_AGGREGATION','1')
    self.compilers.CUDAPPFLAGS = oldFlags
    self.popLanguage()
    return

  def configure(self):
    self.configureLibrary()

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional tests needed by CUSP'''
    if not self.thrust.found:
      raise RuntimeError('CUSP support requires the THRUST package\nRerun configure using --with-thrust-dir')
    config.package.Package.configureLibrary(self)
    self.executeTest(self.configurePC)
    return

