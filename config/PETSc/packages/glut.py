import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions         = ['glutSetWindow']
    self.includes          = ['GLUT/glut.h']
    self.liblist           = [['-framework glut']]
    self.complex           = 1   # 0 means cannot use complex
    self.lookforbydefault  = 0
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.opengl  = framework.require('PETSc.packages.opengl',self)
    self.deps = [self.opengl]
    return

  def getSearchDirectories(self):
    yield ''
    return
