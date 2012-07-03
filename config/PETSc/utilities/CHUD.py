#!/usr/bin/env python
from __future__ import generators
import user
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.types         = framework.require('config.types', self)
    self.languages     = framework.require('PETSc.utilities.languages', self)
    self.setCompilers  = framework.require('config.setCompilers',      self)
    return

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc','-with-chud=<bool>',       nargs.ArgBool(None, 0,'On Apple builds with CHUD profiling (broken)'))


  def configureCHUD(self):
    '''Determines if the Apple CHUD hardware monitoring utilities are available'''
    '''The needed header files are NOT part of Xcode and MUST be downloaded seperately.'''
    self.CPPFLAGS = ' '
    self.LIBS     = ' '
    if config.setCompilers.Configure.isDarwin() and self.framework.argDB['with-chud']:
      self.setCompilers.pushLanguage('C')
      oldFlags = self.setCompilers.CPPFLAGS
      oldLibs  = self.setCompilers.LIBS
      # setting the values here should work, but for some reason they disappear from the variables before
      # PETSc.Configure prints them out
      self.setCompilers.CPPFLAGS += ' -F/System/Library/PrivateFrameworks '
      self.setCompilers.LIBS += ' -framework CHUD -F/System/Library/PrivateFrameworks '
      if self.checkLink('#include <CHUD/CHUD.h>\n', 'int status = chudInitialize();\n'):
        self.haveCHUD = 1
        self.addDefine('HAVE_CHUD', 1)
        self.CPPFLAGS = ' -F/System/Library/PrivateFrameworks '
        self.LIBS     = ' -framework CHUD -F/System/Library/PrivateFrameworks '
        self.framework.logPrint('Located CHUD include files')
      else:
        self.setCompilers.CPPFLAGS = oldFlags
        self.setCompilers.LIBS = oldLibs
      self.setCompilers.popLanguage()
    return

  def configure(self):
    self.executeTest(self.configureCHUD)
    return
