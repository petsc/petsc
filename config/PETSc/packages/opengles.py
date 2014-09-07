import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions         = []
    self.includes          = []
    self.liblist           = [[]]
    self.lookforbydefault  = 0 
    self.worksonWindows    = 0  # 1 means that package can be used on Microsof Windows
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    return

  def configureLibrary(self):
    self.addDefine('HAVE_OPENGL', 1)
    self.addDefine('HAVE_OPENGLES', 1)
