#!/usr/bin/env python
from __future__ import generators
import user
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers',     self)
    self.arch         = self.framework.require('PETSc.utilities.arch', self)
    self.useShared    = 0
    return

  def __str__(self):
    return ''
    
  def setupHelp(self, help):
    return

  def configureSharedLibraries(self):
    if not self.arch.hostOsBase.startswith('aix') and not self.arch.hostOsBase.startswith('darwin'):
      self.useShared = self.framework.argDB['with-shared']
    if self.useShared:
      self.addMakeRule('shared_arch','shared_'+self.arch.hostOsBase)
    else:
      self.addMakeRule('shared','')

  def configure(self):
    self.executeTest(self.configureSharedLibraries)
    return
