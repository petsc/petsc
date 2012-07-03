#!/usr/bin/env python
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
    help.addArgument('PETSc', '-with-errorchecking=<bool>', nargs.ArgBool(None, 1, 'Specify error checking/exceptions in libraries'))    
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    return

  def configureDebugging(self):
    # should do error checking
    if self.framework.argDB['with-errorchecking']:
      self.addDefine('USE_ERRORCHECKING',1)
    else:
      self.logPrintBox('     WARNING! Compiling PETSc with NO error checking/exception handling, \n \
                    this should only be done for timing and production runs where you DO NOT \n \
                    use PETSc exceptions. All development should be done when configured using \n \
                    --with-errorchecking=1')          

    self.debugging = self.compilerFlags.debugging
    if not self.debugging:
      self.logPrintBox('          WARNING! Compiling PETSc with no debugging, this should \n \
               only be done for timing and production runs. All development should \n \
               be done when configured using --with-debugging=1')

  def configure(self):
    self.executeTest(self.configureDebugging)
    return
