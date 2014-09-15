# Temporary package file to enable PCBDDC development components.

import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions         = 0
    self.includes          = 0
    self.liblist           = 0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.deps    = []
    return

  def configureLibrary(self):
    self.found = 1
    self.framework.packages.append(self)
