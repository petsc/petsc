from __future__ import generators
import config.base
from functools import reduce

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
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.compilers = framework.require('config.compilers', self)
    self.functions = framework.require('config.functions', self)
    self.headers   = framework.require('config.headers', self)
    return

  def configureFPTrap(self):
    '''Checking the handling of floating point traps'''
    self.headers.saveLog()
    self.functions.saveLog()
    if self.headers.check('sigfpe.h'):
      if self.functions.check('handle_sigfpes', libraries = 'fpe'):
        self.addDefine('HAVE_IRIX_STYLE_FPTRAP', 1)
        self.compilers.LIBS = '-lfpe '+self.compilers.LIBS
    elif self.headers.check('fpxcp.h') and self.headers.check('fptrap.h'):
      if reduce(lambda x,y: x and y, map(self.functions.check, ['fp_sh_trap_info', 'fp_trap', 'fp_enable', 'fp_disable'])):
        self.addDefine('HAVE_RS6000_STYLE_FPTRAP', 1)
    elif self.headers.check('floatingpoint.h'):
      if self.functions.check('ieee_flags') and self.functions.check('ieee_handler'):
        if self.headers.check('sunmath.h'):
          self.addDefine('HAVE_SOLARIS_STYLE_FPTRAP', 1)
        else:
          self.addDefine('HAVE_SUN4_STYLE_FPTRAP', 1)
    self.logWrite(self.headers.restoreLog())
    self.logWrite(self.functions.restoreLog())
    return

  def configure(self):
    self.executeTest(self.configureFPTrap)
    return
