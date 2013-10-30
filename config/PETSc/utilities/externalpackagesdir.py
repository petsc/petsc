import config.base
import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    return

  def setupDependencies(self, framework):
    self.installdir    = framework.require('PETSc.utilities.installDir',self)
    return

  def configureExternalPackagesDir(self):
    if self.framework.externalPackagesDir is None:
      self.dir = os.path.join(self.installdir.dir, 'externalpackages')
    else:
      self.dir = self.framework.externalPackagesDir
    return

  def configure(self):
    self.executeTest(self.configureExternalPackagesDir)
    return
