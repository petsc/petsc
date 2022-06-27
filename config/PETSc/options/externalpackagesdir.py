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
    '''Set location where external packages will be downloaded to'''
    if self.framework.externalPackagesDir is None:
      self.dir = os.path.join(os.path.abspath(os.path.join(self.arch.arch)), 'externalpackages')
    else:
      self.dir = os.path.join(self.framework.externalPackagesDir,self.arch.arch)
    return

  def cleanExternalpackagesDir(self):
    '''Remove all downloaded external packages, from --with-clean'''
    import shutil
    if self.framework.argDB['with-clean'] and os.path.isdir(self.dir):
      self.logPrintWarning('"with-clean" is specified. Removing all externalpackage files from '+ self.dir)
      shutil.rmtree(self.dir)
    return

  def configure(self):
    self.executeTest(self.setExternalPackagesDir)
    self.executeTest(self.cleanExternalpackagesDir)
    return
