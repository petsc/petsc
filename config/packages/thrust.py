from __future__ import generators
import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.includes        = ['thrust/version.h']
    self.includedir      = ''
    self.forceLanguage   = 'CUDA'
    self.cxx             = 0
    self.archIndependent = 1
    self.worksonWindows  = 1
    return

  def getSearchDirectories(self):
    import os
    yield ''
    yield os.path.join('/usr','local','cuda')
    yield os.path.join('/usr','local','cuda','thrust')
    return

  def configure(self):
    self.configureLibrary()

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional tests needed by Thrust'''
    config.package.Package.configureLibrary(self)
    return

