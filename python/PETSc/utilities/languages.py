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
    if not hasattr(self, 'clanguage'):
      return ''
    return '  Clanguage: ' + self.clanguage +'\n'
    
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-clanguage=<C or C++>', nargs.Arg(None, 'C', 'Specify C or C++ language'))
    help.addArgument('PETSc', '-with-c++-support', nargs.Arg(None, 0, 'When building C, compile C++ portions of external libraries (e.g. Prometheus)'))
    help.addArgument('PETSc', '-with-c-support', nargs.Arg(None, 0, 'When building with C++, compile so may be used directly from C'))
    help.addArgument('PETSc', '-with-fortran', nargs.ArgBool(None, 1, 'Create and install the Fortran wrappers'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    return

  def packagesHaveCxx(self):
    if 'download-prometheus' in self.framework.argDB and self.framework.argDB['download-prometheus']:
      return 1
    if 'download-hypre' in self.framework.argDB and self.framework.argDB['download-hypre']:
      return 1
    if 'download-ml' in self.framework.argDB and self.framework.argDB['download-ml']:
      return 1
    return 0

  def configureCLanguage(self):
    '''Choose between C and C++ bindings'''
    self.clanguage = self.framework.argDB['with-clanguage'].upper().replace('+','x').replace('X','x')
    if not self.clanguage in ['C', 'Cxx']:
      raise RuntimeError('Invalid C language specified: '+str(self.clanguage))

  def configureLanguageSupport(self):
    '''Check c-support c++-support and other misc tests'''
    if self.clanguage == 'C' and not self.framework.argDB['with-c++-support'] and not self.packagesHaveCxx():
      self.framework.argDB['with-cxx'] = '0'
      self.framework.logPrint('Turning off C++ support')
    if self.clanguage == 'Cxx' and self.framework.argDB['with-c-support']:
      if self.scalartype == 'complex':
        raise RuntimeError('Cannot use --with-c-support and --with-scalar-type=complex together')
      self.cSupport = 1
      self.addDefine('USE_EXTERN_CXX', '1')
      self.framework.logPrint('Turning off C++ name mangling')
    else:
      self.cSupport = 0
      self.framework.logPrint('Allowing C++ name mangling')
    self.framework.logPrint('C language is '+str(self.clanguage))
    self.addDefine('CLANGUAGE_'+self.clanguage.upper(),'1')
    self.framework.require('config.setCompilers', None).mainLanguage = self.clanguage
    return

  def configureExternC(self):
    '''Protect C bindings from C++ mangling'''
    if self.clanguage == 'C':
      self.addDefine('USE_EXTERN_CXX',' ')
    return

  def configureFortranLanguage(self):
    '''Turn on Fortran bindings'''
    if not self.framework.argDB['with-fortran']:
      self.framework.argDB['with-fc'] = '0'
      self.framework.logPrint('Not using Fortran')
    else:
      self.framework.logPrint('Using Fortran')
    return

  def configure(self):
    self.executeTest(self.configureCLanguage)
    self.executeTest(self.configureLanguageSupport)
    self.executeTest(self.configureExternC)
    self.executeTest(self.configureFortranLanguage)
    return
