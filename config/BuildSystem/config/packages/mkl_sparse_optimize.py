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

  def checksSupportBaijCrossCase(self):
    '''Determines if MKL Sparse BLAS create routine returns correct status with zero based indexing and columnMajor block layout'''
    includes = '''#include <sys/types.h>\n#if STDC_HEADERS\n#include <stdlib.h>\n#include <stdio.h>\n#include <stddef.h>\n#endif\n#include <mkl.h>'''
    body     = '''sparse_matrix_t A;\n
                  int n=1,ia[1],ja[1];\n
                  float a[1];
                  sparse_status_t status = mkl_sparse_s_create_bsr(&A,SPARSE_INDEX_BASE_ZERO,SPARSE_LAYOUT_COLUMN_MAJOR,n,n,n,ia,ia,ja,a);\n
                  fprintf(output, "  '--known-mklspblas-supports-zero-based=%d',\\n",(status != SPARSE_STATUS_NOT_SUPPORTED));\n
                  '''
    temp1 = self.compilers.LIBS
    temp2 = self.compilers.CPPFLAGS
    self.compilers.LIBS = self.libraries.toString(self.dlib)+' -lm '+self.compilers.LIBS
    self.compilers.CPPFLAGS += ' '+self.headers.toString(self.dinclude)
    result = self.blasLapack.runTimeTest('known-mklspblas-supports-zero-based',includes,body,self.dlib)
    self.compilers.LIBS = temp1
    self.compilers.CPPFLAGS = temp2
    if result:
      self.log.write('Checking for MKL spblas supports zero based indexing: result ' +str(result)+'\n')
      result = int(result)
      if result:
        self.addDefine('MKL_SUPPORTS_BAIJ_ZERO_BASED', 1)
        self.log.write('Found MKL spblas supports zero based indexing: result\n')

  def checkHaveUsableSp2m(self):
    sp2m_test = '#include <mkl_spblas.h>\nsparse_request_t request = SPARSE_STAGE_FULL_MULT_NO_VAL;\n'
    result = self.checkCompile(sp2m_test)
    self.log.write('Looking for mkl_sparse_sp2m() that is usable for MatMatMultSymbolic()/Numeric(): result ' + str(int(result)) + '\n')
    if result:
      # We append _FEATURE at the end to avoid confusion, because some versions of MKL have mkl_sparse_sp2m(), but not a version
      # that is usable for implementing MatMatMultSymbolic()/Numeric().
      self.addDefine('HAVE_MKL_SPARSE_SP2M_FEATURE', 1)

  def checkMklSpblasDeprecated(self):
    # Some versions of MKL use MKL_DEPRECATED and others just use DEPRECATED, so we must check for both.
    deprecated_test1='#include <mkl_spblas.h>\nDEPRECATED void foo();\n'
    deprecated_test2='#include <mkl_spblas.h>\nMKL_DEPRECATED void foo();\n'
    result1 = self.checkCompile(deprecated_test1)
    result2 = self.checkCompile(deprecated_test2)
    result = result1 or result2
    self.log.write('Checking to see if original MKL SpBLAS is declared deprecated: result ' + str(int(result)) + '\n')
    if result:
      self.addDefine('MKL_SPBLAS_DEPRECATED', 1)


  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    if self.found:
      self.executeTest(self.checksSupportBaijCrossCase)
      self.executeTest(self.checkHaveUsableSp2m)
      self.executeTest(self.checkMklSpblasDeprecated)

