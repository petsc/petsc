from __future__ import generators
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix   = ''
    self.substPrefix    = ''
    self.have__float128 = 0
    return

  def __str1__(self):
    output  = '  Scalar type: ' + self.scalartype + '\n'
    output += '  Precision: ' + self.precision + '\n'
    if self.have__float128 and not self.precision == '__float128': output += '  Support for __float128\n'
    return output

  def setupHelp(self, help):
    import nargs
    #  Dec 2016, the __fp16 type is only available with GNU compilers on ARM systems
    help.addArgument('PETSc', '-with-precision=<__fp16,single,double,__float128>', nargs.Arg(None, 'double', 'Specify numerical precision'))
    help.addArgument('PETSc', '-with-scalar-type=<real or complex>', nargs.Arg(None, 'real', 'Specify real or complex numbers'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.types         = framework.require('config.types', self)
    self.languages     = framework.require('PETSc.options.languages', self)
    self.compilers     = framework.require('config.compilers', self)
    self.libraries     = framework.require('config.libraries',self)
    self.setCompilers  = framework.require('config.setCompilers', self)
    self.compilerFlags = framework.require('config.compilerFlags', self)
    self.types         = framework.require('config.types', self)
    self.headers       = framework.require('config.headers', self)
    self.libraries     = framework.require('config.libraries', self)
    return

  def configureScalarType(self):
    '''Choose between real and complex numbers'''
    self.scalartype = self.framework.argDB['with-scalar-type'].lower()
    if self.scalartype == 'complex':
      self.addDefine('USE_COMPLEX', '1')
      if self.languages.clanguage == 'C' and not self.types.c99_complex:
        raise RuntimeError('C Compiler provided does not support C99 complex')
      if self.languages.clanguage == 'Cxx' and not self.types.cxx_complex:
        raise RuntimeError('Cxx compiler provided does not support std::complex')
      if self.languages.clanguage == 'Cxx':
        self.addDefine('USE_CXXCOMPLEX',1)
    elif not self.scalartype == 'real':
      raise RuntimeError('--with-scalar-type must be real or complex')
    self.logPrint('Scalar type is '+str(self.scalartype))
    # On apple isinf() and isnan() do not work when <complex> is included
    self.pushLanguage(self.languages.clanguage)
    if self.scalartype == 'complex' and self.languages.clanguage == 'Cxx':
      if self.checkLink('#include <math.h>\n#include <complex>\n','double b = 2.0;int a = isnormal(b);\n'):
        self.addDefine('HAVE_ISNORMAL',1)
      if self.checkLink('#include <math.h>\n#include <complex>\n','double b = 2.0;int a = isnan(b);\n'):
        self.addDefine('HAVE_ISNAN',1)
      elif self.checkLink('#include <float.h>\n#include <complex>\n','double b = 2.0;int a = _isnan(b);\n'):
        self.addDefine('HAVE__ISNAN',1)
      if self.checkLink('#include <math.h>\n#include <complex>\n','double b = 2.0;int a = isinf(b);\n'):
        self.addDefine('HAVE_ISINF',1)
      elif self.checkLink('#include <float.h>\n#include <complex>\n','double b = 2.0;int a = _finite(b);\n'):
        self.addDefine('HAVE__FINITE',1)
    else:
      if self.checkLink('#include <math.h>\n','double b = 2.0; int a = isnormal(b);\n'):
        self.addDefine('HAVE_ISNORMAL',1)
      if self.checkLink('#include <math.h>\n','double b = 2.0; int a = isnan(b);\n'):
        self.addDefine('HAVE_ISNAN',1)
      elif self.checkLink('#include <float.h>\n','double b = 2.0;int a = _isnan(b);\n'):
        self.addDefine('HAVE__ISNAN',1)
      if self.checkLink('#include <math.h>\n','double b = 2.0; int a = isinf(b);\n'):
        self.addDefine('HAVE_ISINF',1)
      elif self.checkLink('#include <float.h>\n','double b = 2.0;int a = _finite(b);\n'):
        self.addDefine('HAVE__FINITE',1)
    self.popLanguage()
    return

  def configurePrecision(self):
    '''Set the default real number precision for PETSc objects'''
    self.log.write('Checking C compiler works with __float128\n')
    self.have__float128 = 0
    if self.libraries.check('quadmath','logq',prototype='#include <quadmath.h>',call='__float128 f; logq(f);'):
      self.log.write('C compiler with quadmath library\n')
      self.have__float128 = 1
      if hasattr(self.compilers, 'FC'):
        self.libraries.pushLanguage('FC')
        self.log.write('Checking Fortran works with quadmath library\n')
        if self.libraries.check('quadmath','     ',call = '      real*16 s,w; w = 2.0 ;s = cos(w)'):
          self.log.write('Fortran works with quadmath library\n')
        else:
          self.have__float128 = 0
          self.log.write('Fortran fails with quadmath library\n')
        self.libraries.popLanguage()
      if self.have__float128:
          self.libraries.add('quadmath','logq',prototype='#include <quadmath.h>',call='__float128 f; logq(f);')
          self.addDefine('HAVE_REAL___FLOAT128', '1')

    self.precision = self.framework.argDB['with-precision'].lower()
    if self.precision == '__fp16':  # supported by gcc trunk
      if self.scalartype == 'complex':
        raise RuntimeError('__fp16 can only be used with real numbers, not complex')
      if hasattr(self.compilers, 'FC'):
        raise RuntimeError('__fp16 can only be used with C compiler, not Fortran')
      self.addDefine('USE_REAL___FP16', '1')
      self.addMakeMacro('PETSC_SCALAR_SIZE', '16')
    elif self.precision == 'single':
      self.addDefine('USE_REAL_SINGLE', '1')
      self.addMakeMacro('PETSC_SCALAR_SIZE', '32')
    elif self.precision == 'double':
      self.addDefine('USE_REAL_DOUBLE', '1')
      self.addMakeMacro('PETSC_SCALAR_SIZE', '64')
    elif self.precision == '__float128':  # supported by gcc 4.6/gfortran and later
      if self.have__float128:
        self.addDefine('USE_REAL___FLOAT128', '1')
        self.addMakeMacro('PETSC_SCALAR_SIZE', '128')
      else:
        raise RuntimeError('__float128 support not found. --with-precision=__float128 works with gcc-4.6 and newer compilers.')
    else:
      raise RuntimeError('--with-precision must be __fp16, single, double, or __float128')
    self.logPrint('Precision is '+str(self.precision))
    return

  def configure(self):
    self.executeTest(self.configureScalarType)
    self.executeTest(self.configurePrecision)
    return
