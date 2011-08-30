import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions     = ['picture_ximage2asimage']
    self.includes      = ['afterimage.h']
    self.liblist       = [['libAfterImage.a']]
    self.includedir    = ''
    self.libdir        = '../../lib'
    self.double        = 0

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.x11             = framework.require('PETSc.packages.X11', self)
    self.deps = [self.x11]
    return

  def getSearchDirectories(self):
    import os
    yield ''
    yield os.path.join('/usr','local','include','libAfterImage')
    return



