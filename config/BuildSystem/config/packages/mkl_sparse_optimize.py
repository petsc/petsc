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

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    if self.found:
      self.executeTest(self.checksSupportBaijCrossCase)

