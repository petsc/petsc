from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.includes        = ['thrust/version.h']
    self.includedir      = ''
    self.forceLanguage   = 'CUDA'
    self.cxx             = 0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.cuda = framework.require('packages.cuda', self)
    self.deps = [self.cuda]
    return

  def getSearchDirectories(self):
    import os
    yield ''
    yield self.cuda.directory
    yield os.path.join('/usr','local','cuda')
    yield os.path.join('/usr','local','cuda','thrust')
    return

  def configure(self):
    if self.cuda.found:
      self.framework.argDB['with-'+self.package] = 1
    else:
      raise RuntimeError('thrust needs CUDA --- configure with --with-cuda=1')
    config.package.Package.configure(self)

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional tests needed by Thrust'''
    config.package.Package.configureLibrary(self)
    return

