import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions         = ['glutSetWindow']
    self.includes          = ['GLUT/glut.h']
    self.liblist           = [['-framework glut']]
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.opengl  = framework.require('config.packages.opengl',self)
    self.deps = [self.opengl]
    return
