import PETSc.package

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions     = ['IsMagickInstantiated']
    self.includes      = ['MagicCore/MagickCore.h']
    self.liblist       = [['libMagicCore.a']]





