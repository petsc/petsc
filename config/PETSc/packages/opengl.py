import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions         = ['glFlush']
    self.includes          = ['OpenGL/gl.h']
    self.liblist           = [['-framework opengl']]
    self.lookforbydefault  = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.deps = []
    return

  def getSearchDirectories(self):
    yield ''
    return
