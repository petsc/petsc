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

  def configureLanguage(self):
    self.precision  = self.framework.argDB['with-precision']
    if self.precision.lower() == 'single':
      self.addDefine('USE_SINGLE','1')
    elif self.precision.lower() == 'matsingle':
      self.addDefine('USE_MAT_SINGLE','1')
    elif not self.precision.lower() == 'double':
      raise RuntimeError('--with-precision must be single, double or matsingle')

    self.scalartype = self.framework.argDB['with-scalar-type']
    # make sure the language is cxx for complex
    if self.scalartype.lower() == 'complex':
      self.framework.argDB['with-language']='cxx'
      self.addDefine('USE_COMPLEX','1')
    elif not self.scalartype.lower() == 'real':
      raise RuntimeError('--with-scalar-type must be real or complex')

    # should do error checking
    self.language   = self.framework.argDB['with-language'].upper().replace('+','x').replace('X','x')
      
  def configure(self):
    self.executeTest(self.configureLanguage)
    return
