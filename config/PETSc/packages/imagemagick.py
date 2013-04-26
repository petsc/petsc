import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions     = ['IsMagickInstantiated']
    self.includes      = ['magick/MagickCore.h']
    self.liblist       = [['libMagickCore.a']]





