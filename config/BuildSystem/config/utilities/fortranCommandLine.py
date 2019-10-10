#!/usr/bin/env python
from __future__ import generators
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
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.compilers    = self.framework.require('config.compilers', self)
    self.setCompilers = self.framework.require('config.setCompilers', self)
    self.functions    = self.framework.require('config.functions', self)
    self.libraries    = framework.require('config.libraries',  self)
    return

  def configureFortranCommandLine(self):
    '''Check for the mechanism to retrieve command line arguments in Fortran'''

    # These are for when the routines are called from Fortran

    self.libraries.pushLanguage('FC')
    self.libraries.saveLog()
    if self.libraries.check('','', call = '      integer i\n      character*(80) arg\n       i = command_argument_count()\n       call get_command_argument(i,arg)'):
      self.logWrite(self.libraries.restoreLog())
      self.libraries.popLanguage()
      self.addDefine('HAVE_FORTRAN_GET_COMMAND_ARGUMENT',1)
      return

    self.libraries.pushLanguage('FC')
    self.libraries.saveLog()
    if self.libraries.check('','', call = '      integer i\n      character*(80) arg\n       call getarg(i,arg)'):
      self.logWrite(self.libraries.restoreLog())
      self.libraries.popLanguage()
      self.addDefine('HAVE_FORTRAN_GETARG',1)
      return

    # These are for when the routines are called from C
    # We should unify the naming conventions of these.
    self.pushLanguage('C')
    self.libraries.saveLog()
    self.functions.saveLog()
    if self.libraries.check('','getarg', otherLibs = self.compilers.flibs, fortranMangle = 1):
      self.logWrite(self.functions.restoreLog())
      self.logWrite(self.libraries.restoreLog())
      self.popLanguage()
      self.addDefine('HAVE_GETARG',1)
      return

    self.pushLanguage('C')
    self.libraries.saveLog()
    self.functions.saveLog()
    if self.functions.check('ipxfargc_', libraries = self.compilers.flibs):
      self.logWrite(self.functions.restoreLog())
      self.logWrite(self.libraries.restoreLog())
      self.popLanguage()
      self.addDefine('HAVE_PXFGETARG_NEW',1)
      return

    self.pushLanguage('C')
    self.libraries.saveLog()
    self.functions.saveLog()
    if self.functions.check('f90_unix_MP_iargc', libraries = self.compilers.flibs):
      self.logWrite(self.functions.restoreLog())
      self.logWrite(self.libraries.restoreLog())
      self.popLanguage()
      self.addDefine('HAVE_NAGF90',1)
      return

    self.pushLanguage('C')
    self.libraries.saveLog()
    self.functions.saveLog()
    if self.functions.check('PXFGETARG', libraries = self.compilers.flibs):
      self.logWrite(self.functions.restoreLog())
      self.logWrite(self.libraries.restoreLog())
      self.popLanguage()
      self.addDefine('HAVE_PXFGETARG',1)
      return

    self.pushLanguage('C')
    self.libraries.saveLog()
    self.functions.saveLog()
    if self.functions.check('iargc_', libraries = self.compilers.flibs):
      self.logWrite(self.functions.restoreLog())
      self.logWrite(self.libraries.restoreLog())
      self.popLanguage()
      self.addDefine('HAVE_BGL_IARGC',1)
      return

    self.pushLanguage('C')
    self.libraries.saveLog()
    self.functions.saveLog()
    if self.functions.check('GETARG@16', libraries = self.compilers.flibs):
      self.logWrite(self.functions.restoreLog())
      self.logWrite(self.libraries.restoreLog())
      self.popLanguage()
      self.addDefine('USE_NARGS',1)
      self.addDefine('HAVE_IARG_COUNT_PROGNAME',1)
      return

    self.pushLanguage('C')
    self.libraries.saveLog()
    self.functions.saveLog()
    self.functions.check('_gfortran_iargc', libraries = self.compilers.flibs)
    self.logWrite(self.functions.restoreLog())
    self.logWrite(self.libraries.restoreLog())
    self.popLanguage()
    return

  def configure(self):
    if hasattr(self.setCompilers, 'FC'):
      self.executeTest(self.configureFortranCommandLine)
    return
