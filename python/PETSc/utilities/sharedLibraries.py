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
    import nargs
    help.addArgument('PETSc', '-with-shared=<bool>',           nargs.ArgBool(None, 1, 'Build shared libraries for PETSc'))
    return

  def configureSharedLibraries(self):
    if not self.arch.archBase.startswith('aix') and not self.arch.archBase.startswith('darwin'):
      self.useShared = self.framework.argDB['with-shared']
    if self.useShared:
      self.addMakeRule('shared_arch','shared_'+self.arch.archBase)
    else:
      self.addMakeRule('shared','')

  def configure(self):
    self.executeTest(self.configureSharedLibraries)
    return
