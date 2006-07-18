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
    if not hasattr(self, 'scalartype'):
      return ''
    return '  Scalar type:' + self.scalartype + '\n'
    
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-precision=<single,double,longdouble,int,matsingle>', nargs.Arg(None, 'double', 'Specify numerical precision'))    
    help.addArgument('PETSc', '-with-scalar-type=<real or complex>', nargs.Arg(None, 'real', 'Specify real or complex numbers'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.types     = framework.require('config.types', self)
    self.languages = framework.require('PETSc.utilities.languages', self)
    return

  def configureScalarType(self):
    '''Choose between real and complex numbers'''
    self.scalartype = self.framework.argDB['with-scalar-type'].lower()
    if self.scalartype == 'complex':
      self.addDefine('USE_COMPLEX', '1')
      if self.languages.clanguage == 'C' and not self.types.c99_complex:
        raise RuntimeError('C Compiler provided doest not support C99 complex')
      if self.languages.clanguage == 'Cxx' and not self.types.cxx_complex:
        raise RuntimeError('Cxx compiler provided does not support std::complex')
    elif not self.scalartype == 'real':
      raise RuntimeError('--with-scalar-type must be real or complex')
    self.framework.logPrint('Scalar type is '+str(self.scalartype))
    return

  def configurePrecision(self):
    '''Set the default real number precision for PETSc objects'''
    self.precision = self.framework.argDB['with-precision'].lower()
    if self.precision == 'single':
      self.addDefine('USE_SINGLE', '1')
    elif self.precision == 'matsingle':
      self.addDefine('USE_MAT_SINGLE', '1')
    elif self.precision == 'longdouble':
      self.addDefine('USE_LONG_DOUBLE', '1')
    elif self.precision == 'int':
      self.addDefine('USE_INT', '1')
    elif not self.precision == 'double':
      raise RuntimeError('--with-precision must be single, double, longdouble, int or matsingle')
    self.framework.logPrint('Precision is '+str(self.precision))
    return

  def configure(self):
    self.executeTest(self.configureScalarType)
    self.executeTest(self.configurePrecision)
    return
