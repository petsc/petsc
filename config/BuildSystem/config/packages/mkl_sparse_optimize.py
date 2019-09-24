import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.includes         = ['mkl.h','mkl_spblas.h']
    self.functions        = ['mkl_sparse_optimize','mkl_sparse_s_create_bsr']
    self.liblist          = [[]] # use MKL detected by BlasLapack.py
    self.precisions       = ['single','double']
    self.lookforbydefault = 1
    self.requires32bitint = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.blasLapack]
    return

  def checkHaveUsableSp2m(self):
    sp2m_test = '#include <mkl_spblas.h>\nsparse_request_t request = SPARSE_STAGE_FULL_MULT_NO_VAL;\n'
    temp2 = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.dinclude)
    result = self.checkCompile(sp2m_test)
    self.compilers.CPPFLAGS = temp2
    self.log.write('Looking for mkl_sparse_sp2m() that is usable for MatMatMultSymbolic()/Numeric(): result ' + str(int(result)) + '\n')
    if result:
      # We append _FEATURE at the end to avoid confusion, because some versions of MKL have mkl_sparse_sp2m(), but not a version
      # that is usable for implementing MatMatMultSymbolic()/Numeric().
      self.addDefine('HAVE_MKL_SPARSE_SP2M_FEATURE', 1)

  def checkMklSpblasDeprecated(self):
    # Some versions of MKL use MKL_DEPRECATED and others just use DEPRECATED, so we must check for both.
    temp2 = self.compilers.CPPFLAGS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.dinclude)
    deprecated_test1='#include <mkl_spblas.h>\nDEPRECATED void foo();\n'
    deprecated_test2='#include <mkl_spblas.h>\nMKL_DEPRECATED void foo();\n'
    result1 = self.checkCompile(deprecated_test1)
    result2 = self.checkCompile(deprecated_test2)
    self.compilers.CPPFLAGS = temp2
    result = result1 or result2
    self.log.write('Checking to see if original MKL SpBLAS is declared deprecated: result ' + str(int(result)) + '\n')
    if result:
      self.addDefine('MKL_SPBLAS_DEPRECATED', 1)


  def configureLibrary(self):
    if not self.blasLapack.mkl: return
    config.package.Package.configureLibrary(self)
    self.usesopenmp = self.blasLapack.usesopenmp
    if self.found:
      self.executeTest(self.checkHaveUsableSp2m)
      self.executeTest(self.checkMklSpblasDeprecated)

