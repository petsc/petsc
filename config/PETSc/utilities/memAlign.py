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

  def __str1__(self):
    if not hasattr(self, 'memalign'):
      return ''
    return '  Memory alignment: ' + self.memalign + '\n'

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-memalign=<4,8,16,32,64>', nargs.Arg(None, '16', 'Specify alignment of arrays allocated by PETSc'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.types     = framework.require('config.types', self)
    self.languages = framework.require('PETSc.utilities.languages', self)
    self.compilers = framework.require('config.compilers', self)
    return

  def configureMemAlign(self):
    '''Choose alignment'''
    # Intel/AMD cache lines are 64 bytes, default page sizes are usually 4kB. It would be pretty silly to want that much alignment by default.
    valid = ['4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048', '4096', '8192']
    self.memalign = self.framework.argDB['with-memalign']
    if self.memalign in valid:
      self.addDefine('MEMALIGN', self.memalign)
    else:
      raise RuntimeError('--with-memalign must be in' + str(valid))
    self.framework.logPrint('Memory alignment is ' + self.memalign)
    return

  def configure(self):
    self.executeTest(self.configureMemAlign)
    return
