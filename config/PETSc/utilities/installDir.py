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
    help.addArgument('PETSc', '-with-clean=<bool>',         nargs.ArgBool(None, 0, 'Delete prior build files including externalpackages'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.arch = framework.require('PETSc.utilities.arch', self)
    return

  def setInstallDir(self):
    ''' setup installDir to PETSC_DIR/PETSC_ARCH'''
    self.dir = os.path.abspath(os.path.join(self.arch.arch))

  def configureInstallDir(self):
    '''Makes  installDir subdirectories if it does not exist'''
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

  def cleanInstallDir(self):
    import shutil
    if self.framework.argDB['with-clean'] and os.path.isdir(self.dir):
      self.logPrintBox('Warning: "with-clean" is specified. Removing all build files from '+ self.dir)
      shutil.rmtree(self.dir)
    return

  def saveReconfigure(self):
    self.reconfigure_file = os.path.join(self.dir,'conf','reconfigure-'+self.arch.arch+'.py')
    self.save_reconfigure_file = None
    if self.framework.argDB['with-clean'] and os.path.exists(self.reconfigure_file):
      self.save_reconfigure_file = '.save.reconfigure-'+self.arch.arch+'.py'
      try:
        if os.path.exists(self.save_reconfigure_file): os.unlink(self.save_reconfigure_file)
        os.rename(self.reconfigure_file,self.save_reconfigure_file)
      except Exception, e:
        self.save_reconfigure_file = None
        self.framework.logPrint('error in saveReconfigure(): '+ str(e))
    return

  def restoreReconfigure(self):
    if self.framework.argDB['with-clean'] and self.save_reconfigure_file:
      try:
        os.rename(self.save_reconfigure_file,self.reconfigure_file)
      except Exception, e:
        self.framework.logPrint('error in restoreReconfigure(): '+ str(e))
    return

  def configure(self):
    self.executeTest(self.setInstallDir)
    self.executeTest(self.saveReconfigure)
    self.executeTest(self.cleanInstallDir)
    self.executeTest(self.configureInstallDir)
    self.executeTest(self.restoreReconfigure)
    return
