#!/usr/bin/env python
from __future__ import generators
import user
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers', self)
    self.functions    = self.framework.require('config.functions', self)
    return

  def __str__(self):
    return ''
    
  def setupHelp(self, help):
    import nargs
    return

  def configureFortranCommandLine(self):
    '''Check for the mechanism to retrieve command line arguments in Fortran'''
    self.pushLanguage('C')
    if self.functions.check('_gfortran_iargc', libraries = self.compilers.flibs):
      self.addDefine('HAVE_GFORTRAN_IARGC',1)
      # this needs to be removed later - as gfortran fixed this to confirm with g77 in CVS
      self.addDefine('HAVE_IARG_COUNT_PROGNAME',1)
    elif self.functions.check('ipxfargc_', libraries = self.compilers.flibs):
      self.addDefine('HAVE_PXFGETARG_NEW',1)
    elif self.functions.check('f90_unix_MP_iargc', libraries = self.compilers.flibs):
      self.addDefine('HAVE_NAGF90',1)
    elif self.functions.check('PXFGETARG', libraries = self.compilers.flibs):
      self.addDefine('HAVE_PXFGETARG',1)
    elif self.functions.check('GETARG@16', libraries = self.compilers.flibs): 
      self.addDefine('USE_NARGS',1)
      self.addDefine('HAVE_IARG_COUNT_PROGNAME',1)
    self.popLanguage()
    return

  def configure(self):
    self.executeTest(self.configureFortranCommandLine)
    return
