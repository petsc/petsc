import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions         = ['glFlush']
    self.includes          = ['OpenGL/gl.h']
    self.liblist           = [['-framework opengl']]
    self.lookforbydefault  = 0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.deps = []
    return
