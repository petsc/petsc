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
    help.addArgument('PETSc', '-with-language=<C or C++>', nargs.Arg(None, 'C', 'Specify C or C++ language'))
    help.addArgument('PETSc', '-with-precision=<single,double,matsingle>', nargs.Arg(None, 'double', 'Specify numerical precision'))    
    help.addArgument('PETSc', '-with-scalar-type=<real or complex>', nargs.Arg(None, 'real', 'Specify real or complex numbers'))
    return

  def configureScalarType(self):
    '''Choose between real and complex numbers'''
    self.scalartype = self.framework.argDB['with-scalar-type'].lower()
    if self.scalartype == 'complex':
      self.framework.argDB['with-language'] = 'Cxx'
      self.addDefine('USE_COMPLEX', '1')
    elif not self.scalartype == 'real':
      raise RuntimeError('--with-scalar-type must be real or complex')
    return

  def configurePrecision(self):
    '''Set the default real number precision for PETSc objects'''
    self.precision = self.framework.argDB['with-precision'].lower()
    if self.precision == 'single':
      self.addDefine('USE_SINGLE', '1')
    elif self.precision == 'matsingle':
      self.addDefine('USE_MAT_SINGLE', '1')
    elif not self.precision == 'double':
      raise RuntimeError('--with-precision must be single, double, or matsingle')
    return

  def configureLanguage(self):
    '''Choose between C and C++ bindings'''
    self.language = self.framework.argDB['with-language'].upper().replace('+','x').replace('X','x')
    if not self.language in ['C', 'Cxx']:
      raise RuntimeError('Invalid language specified: '+str(self.language))
    return

  def configureExternC(self):
    '''Protect C bindings from C++ mangling'''
    if self.language == 'C':
      self.addDefine('USE_EXTERN_CXX')
    return

  def configure(self):
    self.executeTest(self.configureScalarType)
    self.executeTest(self.configurePrecision)
    self.executeTest(self.configureLanguage)
    self.executeTest(self.configureExternC)
    return
