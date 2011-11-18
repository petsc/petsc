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

  def __str__(self):
    return ''
    
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-cmake=<prog>', nargs.Arg(None, 'cmake', 'Specify cmake'))
    return

  def configureCMake(self):
    '''Check various things about cmake'''
    self.found = self.getExecutable(self.framework.argDB['with-cmake'], getFullPath = 1,resultName = 'cmake')
    return

  def configure(self):
    self.executeTest(self.configureCMake)
    return
