#!/usr/bin/env python
from __future__ import generators
import config.base
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.dir = ''
    return

  def __str__(self):
    return ''

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-clean=<bool>',         nargs.ArgBool(None, 0, 'Delete prior build files including externalpackages'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.petscdir = framework.require('PETSc.options.petscdir', self)
    self.arch     = framework.require('PETSc.options.arch', self)
    return

  def printSudoPasswordMessage(self,needsudo = 1):
    '''Prints a message that sudo password will be needed for installs of packages'''
    '''Packages like sowing and make that are never installed in an sudo location would pass 0 for needsudo'''
    if needsudo and self.installSudoMessage:
      self.logPrintBox(self.installSudoMessage)
      self.installSudoMessage = ''

  def setInstallDir(self):
    ''' setup installDir to either prefix or if that is not set to PETSC_DIR/PETSC_ARCH'''
    self.installSudo        = ''
    self.installSudoMessage = ''
    if self.framework.argDB['prefix']:
      self.dir = os.path.abspath(os.path.expanduser(self.framework.argDB['prefix']))
      self.petscDir = self.dir
      self.petscArch = ''
      try:
        os.makedirs(os.path.join(self.dir,'PETScTestDirectory'))
        os.rmdir(os.path.join(self.dir,'PETScTestDirectory'))
      except Exception as e:
        self.logPrint('Error trying to to test write permissions on directory '+str(e))
        self.installSudoMessage = 'You do not have write permissions to the --prefix directory '+self.dir+'\nYou will be prompted for the sudo password for any external package installs'
        self.installSudo = 'sudo '
    else:
      self.dir = os.path.abspath(os.path.join(self.petscdir.dir, self.arch.arch))
      self.petscDir = self.petscdir.dir
      self.petscArch = self.arch.arch
    self.addMakeMacro('PREFIXDIR',self.dir)
    self.confDir = os.path.abspath(os.path.join(self.petscdir.dir, self.arch.arch))

  def configureInstallDir(self):
    '''Makes  installDir subdirectories if it does not exist for both prefix install location and PETSc work install location'''
    dir = os.path.abspath(os.path.join(self.petscdir.dir, self.arch.arch))
    if not os.path.exists(dir):
      os.makedirs(dir)
    for i in ['include','lib','bin',os.path.join('lib','petsc','conf')]:
      newdir = os.path.join(dir,i)
      if not os.path.exists(newdir):
        os.makedirs(newdir)
    if os.path.isfile(self.framework.argDB.saveFilename):
      os.remove(self.framework.argDB.saveFilename)
    confdir = os.path.join(dir,'lib','petsc','conf')
    self.framework.argDB.saveFilename = os.path.abspath(os.path.join(confdir, 'RDict.db'))
    self.logPrint('Changed persistence directory to '+confdir)
    return

  def cleanConfDir(self):
    import shutil
    if self.framework.argDB['with-clean'] and os.path.isdir(self.confDir):
      self.logPrintBox('Warning: "with-clean" is specified. Removing all build files from '+ self.confDir)
      shutil.rmtree(self.confDir)
    return

  def saveReconfigure(self):
    self.reconfigure_file = os.path.join(self.dir,'lib','petsc','conf','reconfigure-'+self.arch.arch+'.py')
    self.save_reconfigure_file = None
    if self.framework.argDB['with-clean'] and os.path.exists(self.reconfigure_file):
      self.save_reconfigure_file = '.save.reconfigure-'+self.arch.arch+'.py'
      try:
        if os.path.exists(self.save_reconfigure_file): os.unlink(self.save_reconfigure_file)
        os.rename(self.reconfigure_file,self.save_reconfigure_file)
      except Exception as e:
        self.save_reconfigure_file = None
        self.logPrint('error in saveReconfigure(): '+ str(e))
    return

  def restoreReconfigure(self):
    if self.framework.argDB['with-clean'] and self.save_reconfigure_file:
      try:
        os.rename(self.save_reconfigure_file,self.reconfigure_file)
      except Exception as e:
        self.logPrint('error in restoreReconfigure(): '+ str(e))
    return

  def configure(self):
    self.executeTest(self.setInstallDir)
    self.executeTest(self.saveReconfigure)
    self.executeTest(self.cleanConfDir)
    self.executeTest(self.configureInstallDir)
    self.executeTest(self.restoreReconfigure)
    return
