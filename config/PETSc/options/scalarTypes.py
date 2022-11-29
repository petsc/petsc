from __future__ import generators
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix   = ''
    self.substPrefix    = ''
    self.have__fp16     = 0
    self.have__float128 = 0
    return

  def __str1__(self):
    output  = '  Scalar type: ' + self.scalartype + '\n'
    output += '  Precision: ' + self.precision + '\n'
    support = []
    if self.have__fp16 and not self.precision == '__fp16': support.append('__fp16')
    if self.have__float128 and not self.precision == '__float128': support.append('__float128')
    if not len(support) == 0: output += '  Support for ' + ' and '.join(support) + '\n'
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
      if self.checkLink('#include <math.h>\n#include <complex>\n','double b = 2.0;int a = isnormal(b);(void)a'):
        self.addDefine('HAVE_ISNORMAL',1)
      if self.checkLink('#include <math.h>\n#include <complex>\n','double b = 2.0;int a = isnan(b);(void)a'):
        self.addDefine('HAVE_ISNAN',1)
      elif self.checkLink('#include <float.h>\n#include <complex>\n','double b = 2.0;int a = _isnan(b);(void)a'):
        self.addDefine('HAVE__ISNAN',1)
      if self.checkLink('#include <math.h>\n#include <complex>\n','double b = 2.0;int a = isinf(b);(void)a'):
        self.addDefine('HAVE_ISINF',1)
      elif self.checkLink('#include <float.h>\n#include <complex>\n','double b = 2.0;int a = _finite(b);(void)a'):
        self.addDefine('HAVE__FINITE',1)
    else:
      if self.checkLink('#include <math.h>\n','double b = 2.0; int a = isnormal(b);(void)a'):
        self.addDefine('HAVE_ISNORMAL',1)
      if self.checkLink('#include <math.h>\n','double b = 2.0; int a = isnan(b);(void)a'):
        self.addDefine('HAVE_ISNAN',1)
      elif self.checkLink('#include <float.h>\n','double b = 2.0;int a = _isnan(b);(void)a'):
        self.addDefine('HAVE__ISNAN',1)
      if self.checkLink('#include <math.h>\n','double b = 2.0; int a = isinf(b);(void)a'):
        self.addDefine('HAVE_ISINF',1)
      elif self.checkLink('#include <float.h>\n','double b = 2.0;int a = _finite(b);(void)a'):
        self.addDefine('HAVE__FINITE',1)
    self.popLanguage()
    return

  def configurePrecision(self):
    '''Set the default real number precision for PETSc objects'''
    self.log.write('Checking C compiler works with __float128\n')
    self.have__float128 = 0
    if self.libraries.check('quadmath','logq',prototype='#include <quadmath.h>',call='__float128 f = 0.0; logq(f)'):
      self.log.write('C compiler works with quadmath library\n')
      self.have__float128 = 1
      if hasattr(self.compilers, 'FC'):
        self.libraries.pushLanguage('FC')
        self.log.write('Checking Fortran compiler works with quadmath library\n')
        if self.libraries.check('quadmath','     ',call = '      real*16 s,w; w = 2.0; s = cos(w)'):
          self.log.write('Fortran compiler works with quadmath library\n')
        else:
          self.have__float128 = 0
          self.log.write('Fortran compiler fails with quadmath library\n')
        self.libraries.popLanguage()
      if hasattr(self.compilers, 'CXX'):
        self.libraries.pushLanguage('Cxx')
        isGNU = self.setCompilers.isGNU(self.getCompiler(lang='Cxx'), self.log)
        # need to bypass g++ error: non-standard suffix on floating constant [-Werror=pedantic]
        # this warning can't be disabled but is actually never triggered by PETSc
        if isGNU:
          self.setCompilers.pushLanguage('Cxx')
          preprocessorFlagsArg = self.setCompilers.getPreprocessorFlagsArg()
          oldPreprocessorFlags = getattr(self.setCompilers, preprocessorFlagsArg)
          setattr(self.setCompilers, preprocessorFlagsArg, oldPreprocessorFlags+' -Wno-error')
        self.log.write('Checking C++ compiler works with quadmath library\n')
        if self.libraries.check('quadmath','logq',prototype='#include <quadmath.h>',call='__float128 f = FLT128_EPSILON; logq(f)'):
          self.log.write('C++ compiler works with quadmath library\n')
        else:
          self.have__float128 = 0
          self.log.write('C++ compiler fails with quadmath library\n')
        if isGNU:
          setattr(self.setCompilers, preprocessorFlagsArg, oldPreprocessorFlags)
          self.setCompilers.popLanguage()
        self.libraries.popLanguage()
      if self.have__float128:
          self.libraries.add('quadmath','logq',prototype='#include <quadmath.h>',call='__float128 f = 0.0; logq(f)')
          self.addDefine('HAVE_REAL___FLOAT128', '1')

    self.log.write('Checking C compiler works with __fp16\n')
    self.have__fp16 = 0
    if self.libraries.check('','',call='__fp16 f = 1.0, g; g = ret___fp16(f); (void)g',prototype='static __fp16 ret___fp16(__fp16 f) { return f; }'):
      self.have__fp16 = 1
      self.addDefine('HAVE_REAL___FP16', '1')

    self.precision = self.framework.argDB['with-precision'].lower()
    if self.precision == '__fp16':  # supported by gcc trunk
      if self.scalartype == 'complex':
        raise RuntimeError('__fp16 can only be used with real numbers, not complex')
      if hasattr(self.compilers, 'FC'):
        raise RuntimeError('__fp16 can only be used with C compiler, not Fortran')
      if self.have__fp16:
        self.addDefine('USE_REAL___FP16', '1')
        self.addMakeMacro('PETSC_SCALAR_SIZE', '16')
      else:
        raise RuntimeError('__fp16 support not found, cannot proceed --with-precision=__fp16')
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
