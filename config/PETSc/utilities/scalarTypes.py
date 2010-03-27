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
    desc = []
    if hasattr(self, 'scalartype'):
      desc.append('  Scalar type: ' + self.scalartype)
    if hasattr(self, 'precision'):
      desc.append('  Precision: ' + self.precision)
    return '\n'.join(desc)+'\n'
    
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-precision=<single,double,longdouble,int,matsingle,qd_dd>', nargs.Arg(None, 'double', 'Specify numerical precision'))    
    help.addArgument('PETSc', '-with-scalar-type=<real or complex>', nargs.Arg(None, 'real', 'Specify real or complex numbers'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.types     = framework.require('config.types', self)
    self.languages = framework.require('PETSc.utilities.languages', self)
    self.compilers = framework.require('config.compilers', self)
    self.qd        = framework.require('PETSc.packages.qd',self)
    return


  def configureScalarType(self):
    '''Choose between real and complex numbers'''
    self.scalartype = self.framework.argDB['with-scalar-type'].lower()
    if self.scalartype == 'complex':
      self.addDefine('USE_COMPLEX', '1')
      if self.languages.cSupport:
        raise RuntimeError('Cannot use --with-c-support and --with-scalar-type=complex together')      
      if self.languages.clanguage == 'C' and not self.types.c99_complex:
        raise RuntimeError('C Compiler provided doest not support C99 complex')
      if self.languages.clanguage == 'Cxx' and not self.types.cxx_complex:
        raise RuntimeError('Cxx compiler provided does not support std::complex')
    elif not self.scalartype == 'real':
      raise RuntimeError('--with-scalar-type must be real or complex')
    self.framework.logPrint('Scalar type is '+str(self.scalartype))
    # On apple isinf() and isnan() do not work when <complex> is included
    self.pushLanguage(self.languages.clanguage)
    if self.scalartype == 'complex' and self.languages.clanguage == 'Cxx':
      if self.checkLink('#include <math.h>\n#include <complex>\n','double b = 2.0;int a = isnan(b);\n'):
        self.addDefine('HAVE_ISNAN',1)    
      if self.checkLink('#include <math.h>\n#include <complex>\n','double b = 2.0;int a = isinf(b);\n'):
        self.addDefine('HAVE_ISINF',1)    
      if self.checkLink('#include <math.h>\n#include <complex>\n','double b = 2.0;int a = _isnan(b);\n'):
        self.addDefine('HAVE__ISNAN',1)    
      if self.checkLink('#include <math.h>\n#include <complex>\n','double b = 2.0;int a = _finite(b);\n'):
        self.addDefine('HAVE__FINITE',1)    
    else:
      if self.checkLink('#include <math.h>\n','double b = 2.0; int a = isnan(b);\n'):
        self.addDefine('HAVE_ISNAN',1)    
      if self.checkLink('#include <math.h>\n','double b = 2.0; int a = isinf(b);\n'):
        self.addDefine('HAVE_ISINF',1)    
      if self.checkLink('#include <math.h>\n','double b = 2.0;int a = _isnan(b);\n'):
        self.addDefine('HAVE__ISNAN',1)    
      if self.checkLink('#include <math.h>\n','double b = 2.0;int a = _finite(b);\n'):
        self.addDefine('HAVE__FINITE',1)    
    self.popLanguage()
    return

  def configurePrecision(self):
    '''Set the default real number precision for PETSc objects'''
    self.precision = self.framework.argDB['with-precision'].lower()
    if self.precision == 'single':
      self.addDefine('USE_SCALAR_SINGLE', '1')
    elif self.precision == 'matsingle':
      self.addDefine('USE_SCALAR_MAT_SINGLE', '1')
    elif self.precision == 'longdouble':
      self.pushLanguage('C')
      if config.setCompilers.Configure.isIntel(self.compilers.getCompiler()):
        # Intel's C long double is 80 bits, so does not match Fortran's real*16, but
        # Intel C has a _Quad that is 128 bits
        self.precision = 'quad'
        self.addDefine('USE_SCALAR__QUAD', '1')        
      self.popLanguage()
      self.addDefine('USE_SCALAR_LONG_DOUBLE', '1')
    elif self.precision == 'int':
      self.addDefine('USE_SCALAR_INT', '1')
    elif self.precision == 'qd_dd':
      self.addDefine('USE_SCALAR_QD_DD', '1')
    elif not self.precision == 'double':
      raise RuntimeError('--with-precision must be single, double, longdouble, int, qd_dd or matsingle')
    self.framework.logPrint('Precision is '+str(self.precision))
    return

  def configure(self):
    self.executeTest(self.configureScalarType)
    self.executeTest(self.configurePrecision)
    return
