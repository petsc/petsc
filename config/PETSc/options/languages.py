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
    if not hasattr(self, 'clanguage'):
      return ''
    return '  Clanguage: ' + self.clanguage +'\n'

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-clanguage=<C or C++>', nargs.Arg(None, 'C', 'Specify C (recommended) or C++ to compile PETSc. You can use C++ in either case.'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    return

  def configureCLanguage(self):
    '''Choose whether to compile the PETSc library using a C or C++ compiler'''
    self.clanguage = self.framework.argDB['with-clanguage'].upper().replace('+','x').replace('X','x')
    if not self.clanguage in ['C', 'Cxx']:
      raise RuntimeError('Invalid C language specified: '+str(self.clanguage))
    self.logPrint('C language is '+str(self.clanguage))
    self.addDefine('CLANGUAGE_'+self.clanguage.upper(),'1')
    self.framework.require('config.setCompilers', None).mainLanguage = self.clanguage

  def configure(self):
    self.executeTest(self.configureCLanguage)
    return
