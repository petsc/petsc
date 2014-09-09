import config.package
import os

# This is a temporary file for defining the flag PETSC_THREADCOMM_ACTIVE.
# It should be deleted when the flag is removed from the code.

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.deps    = []
    return

  def configureLibrary(self):
    self.addDefine('THREADCOMM_ACTIVE',1)
    return
