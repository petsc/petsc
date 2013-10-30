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

  def configureInstallDir(self):
    '''Makes $PETSC_ARCH and subdirectories if it does not exist'''
    self.dir = os.path.abspath(os.path.join(self.arch.arch))
    if not os.path.exists(self.dir):
      os.makedirs(self.dir)
    for i in ['include','lib','bin','conf']:
      newdir = os.path.join(self.dir,i)
      if not os.path.exists(newdir):
        os.makedirs(newdir)
    if os.path.isfile(self.framework.argDB.saveFilename):
      os.remove(self.framework.argDB.saveFilename)
    confdir = os.path.join(self.dir,'conf')
    self.framework.argDB.saveFilename = os.path.abspath(os.path.join(confdir, 'RDict.db'))
    self.framework.logPrint('Changed persistence directory to '+confdir)
    return

  def configure(self):
    self.executeTest(self.configureInstallDir)
    return
