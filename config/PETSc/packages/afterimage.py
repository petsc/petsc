import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions     = ['picture_ximage2asimage']
    self.includes      = ['afterimage.h']
    self.liblist       = [['libAfterImage.a']]
    self.includedir    = ''
    self.libdir        = '../../lib'

  def getSearchDirectories(self):
    import os
    yield ''
    yield os.path.join('/usr','local','include','libAfterImage')
    return



