import config.base
import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    return

  def setupDependencies(self, framework):
    self.installdir    = framework.require('PETSc.options.installDir',self)
    self.arch          = framework.require('PETSc.options.arch',self)
    return

  def setExternalPackagesDir(self):
    if self.framework.externalPackagesDir is None:
      self.dir = os.path.join(os.path.abspath(os.path.join(self.arch.arch)), 'externalpackages')
    else:
      self.dir = os.path.join(self.framework.externalPackagesDir,self.arch.arch)
    return

  def cleanExternalpackagesDir(self):
    import shutil
    if self.framework.argDB['with-clean'] and os.path.isdir(self.dir):
      self.logPrintBox('Warning: "with-clean" is specified. Removing all externalpackage files from '+ self.dir)
      shutil.rmtree(self.dir)
    return

  def configure(self):
    self.executeTest(self.setExternalPackagesDir)
    self.executeTest(self.cleanExternalpackagesDir)

    return
