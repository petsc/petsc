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
    help.addArgument('PETSc', '-with-debugging=<yes or no>', nargs.ArgBool(None, 1, 'Specify debugging version of libraries'))
    return

  def configureDebugging(self):
    # should do error checking
    self.debugging = self.framework.argDB['with-debugging']
    if not self.debugging:
      self.framework.logClear()
      print '=================================================================================\r'
      print '          WARNING! Compiling PETSc with no error checking, this should \r'
      print '        only be done for timing and production runs. All development should \r'
      print '        be done when configured with --with-debugging=1 \r'          
      print '=================================================================================\r'  
    
  def configure(self):
    self.executeTest(self.configureDebugging)
    return
