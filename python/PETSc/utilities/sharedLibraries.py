#!/usr/bin/env python
from __future__ import generators
import user
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.useShared    = 0
    return

  def __str__(self):
    return ''
    
  def setupHelp(self, help):
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.arch = framework.require('PETSc.utilities.arch', self)
    self.setCompilers = framework.require('config.setCompilers', self)
    return

  def configureSharedLibraries(self):
    self.useShared = not self.setCompilers.staticLibraries
    if self.useShared:
      self.addMakeRule('shared_arch','shared_'+self.arch.hostOsBase)
      self.addMakeMacro('BUILDSHAREDLIB','yes')
    else:
      self.addMakeRule('shared_arch','')
      self.addMakeMacro('BUILDSHAREDLIB','no')
  def configure(self):
    self.executeTest(self.configureSharedLibraries)
    return
