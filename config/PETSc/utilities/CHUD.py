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
    self.types     = framework.require('config.types', self)
    self.languages = framework.require('PETSc.utilities.languages', self)
    self.compilers = framework.require('config.compilers', self)
    return

  def configureCHUD(self):
    '''Determines if the Apple CHUD hardware monitoring utilities are available'''
    '''The needed header files are NOT part of Xcode and MUST be downloaded seperately.'''
    if config.setCompilers.Configure.isDarwin():
      self.compilers.pushLanguage('C')
      oldFlags = self.compilers.CPPFLAGS
      oldLibs  = self.compilers.LIBS
      self.compilers.CPPFLAGS += ' -F/System/Library/PrivateFrameworks '
      self.compilers.LIBS += ' -framework CHUD -F/System/Library/PrivateFrameworks '
      if self.checkLink('#include <CHUD/CHUD.h>\n', 'int status = chudInitialize();\n'):
        self.haveCHUD = 1
        self.addDefine('HAVE_CHUD', 1)
        self.framework.logPrint('Located CHUD include files')
      else:
        self.compilers.CPPFLAGS = oldFlags
        self.compilers.LIBS = oldLibs
      self.compilers.popLanguage()
    return

  def configure(self):
    self.executeTest(self.configureCHUD)
    return
