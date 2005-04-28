#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def __str__(self):
    return ''
    
  def setupHelp(self, help):
    import nargs
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.arch = framework.require('PETSc.utilities.arch', self)
    return

  def configureBmakeDir(self):
    '''Makes bmake/$PETSC_ARCH if it does not exist'''
    self.bmakeDir = os.path.join('bmake', self.arch.arch)
    if not os.path.exists(self.bmakeDir):
      os.makedirs(self.bmakeDir)
      self.framework.actions.addArgument('PETSc', 'Directory creation', 'Created '+self.bmakeDir+' for configuration data')
    if os.path.isfile(self.framework.argDB.saveFilename):
      os.remove(self.framework.argDB.saveFilename)
    self.framework.argDB.saveFilename = os.path.abspath(os.path.join(self.bmakeDir, 'RDict.db'))
    self.framework.logPrint('Changed persistence directory to '+self.bmakeDir)
    return

  def configure(self):
    self.executeTest(self.configureBmakeDir)
    return
