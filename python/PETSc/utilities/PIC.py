#!/usr/bin/env python
from __future__ import generators
import user
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers',                self)
    self.dynamic      = self.framework.require('PETSc.utilities.dynamicLibraries',self)    
    return

  def __str__(self):
    return ''
    
  def setupHelp(self, help):
    import nargs
    return


  def configurePIC(self):
    '''Determine the PIC option for each compiler
       - There needs to be a test that checks that the functionality is actually working'''
    if not self.dynamic.useDynamic:
      return
    if self.framework.argDB['PETSC_ARCH_BASE'].startswith('hpux') and not config.setCompilers.Configure.isGNU(self.framework.argDB['CC']):
      return
    languages = ['C']
    if 'CXX' in self.framework.argDB:
      languages.append('C++')
    if 'FC' in self.framework.argDB:
      languages.append('F77')
    for language in languages:
      self.pushLanguage(language)
      for testFlag in ['-PIC', '-fPIC', '-KPIC']:
        try:
          self.framework.log.write('Trying '+language+' compiler flag '+testFlag+'\n')
          self.addCompilerFlag(testFlag)
          break
        except RuntimeError:
          self.framework.log.write('Rejected '+language+' compiler flag '+testFlag+'\n')
      self.popLanguage()
    return


  def configure(self):
    self.executeTest(self.configurePIC)
    return
