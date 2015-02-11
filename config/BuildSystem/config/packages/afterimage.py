import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions     = ['picture_ximage2asimage']
    self.includes      = ['afterimage.h']
    self.liblist       = [['libAfterImage.a']]
    self.includedir    = ''
    self.libdir        = '../../lib'

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.x               = framework.require('config.packages.X', self)
    self.deps = [self.x]
    return

  def getSearchDirectories(self):
    import os
    yield ''
    yield os.path.join('/usr','local','include','libAfterImage')
    return



