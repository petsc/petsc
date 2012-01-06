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
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.compilers = self.framework.require('config.compilers', self)
    self.functions = self.framework.require('config.functions', self)
    self.libraries     = framework.require('config.libraries',  self)
    return

  def configureFortranCommandLine(self):
    '''Check for the mechanism to retrieve command line arguments in Fortran'''

    # These are for when the routines are called from Fortran 
    if hasattr(self.compilers, 'FC'):
      self.libraries.pushLanguage('FC')
      if self.libraries.check('','', call = '      integer i\n      character*(80) arg\n       call get_command_argument(i,arg)'):
        self.addDefine('HAVE_FORTRAN_GET_COMMAND_ARGUMENT',1)
      elif self.libraries.check('','', call = '      integer i\n      character*(80) arg\n       call getarg(i,arg)'):
        self.addDefine('HAVE_FORTRAN_GETARG',1)
      self.libraries.popLanguage()

    # These are for when the routines are called fraom C
    # We should unify the naming conventions of these.
    self.pushLanguage('C')
    # This one is not currently used in PETSc source code
    if self.libraries.check('','get_command_argument', otherLibs = self.compilers.flibs, fortranMangle = 1):
      self.addDefine('HAVE_GET_COMMAND_ARGUMENT',1)
    if self.libraries.check('','getarg', otherLibs = self.compilers.flibs, fortranMangle = 1):
      self.addDefine('HAVE_GETARG',1)
    if self.functions.check('ipxfargc_', libraries = self.compilers.flibs):
      self.addDefine('HAVE_PXFGETARG_NEW',1)
    elif self.functions.check('f90_unix_MP_iargc', libraries = self.compilers.flibs):
      self.addDefine('HAVE_NAGF90',1)
    elif self.functions.check('PXFGETARG', libraries = self.compilers.flibs):
      self.addDefine('HAVE_PXFGETARG',1)
    elif self.functions.check('iargc_', libraries = self.compilers.flibs):
      self.addDefine('HAVE_BGL_IARGC',1)
    elif self.functions.check('GETARG@16', libraries = self.compilers.flibs): 
      self.addDefine('USE_NARGS',1)
      self.addDefine('HAVE_IARG_COUNT_PROGNAME',1)
    elif self.functions.check('_gfortran_iargc', libraries = self.compilers.flibs):
      self.addDefine('HAVE_GFORTRAN_IARGC',1)
    self.popLanguage()
    return

  def configure(self):
    self.executeTest(self.configureFortranCommandLine)
    return
