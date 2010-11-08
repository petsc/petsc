import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions = ['cublasInit', 'cufftDestroy']
    self.includes  = ['cublas.h', 'cufft.h']
    self.liblist   = [['libcufft.a', 'libcublas.a','libcudart.a']]
    self.double    = 0   # 1 means requires double precision 
    self.cxx       = 0
    self.requiredVersion = '3.2'
    return

  def setupDependencies(self, framework):
    PETSc.package.NewPackage.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers',self)
    self.headers      = framework.require('config.headers',self)
    self.scalartypes  = framework.require('PETSc.utilities.scalarTypes', self)        
    self.languages    = framework.require('PETSc.utilities.languages',   self)
    self.cusp         = framework.require('config.packages.cusp',        self)
    self.thrust       = framework.require('config.packages.thrust',      self)
    return

  def getSearchDirectories(self):
    import os
    yield ''
    yield os.path.join('/usr','local','cuda')
    return
  
  def checkSizeofVoidP(self):
    '''Checks if the CUDA compiler agrees with the C compiler on what size of void * should be'''
    self.framework.log.write('Checking if sizeof(void*) in CUDA is the same as with regular compiler\n')
    typeName = 'void*'
    filename = 'conftestval'
    includes = '''
#include <sys/types.h>
#if STDC_HEADERS
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#endif\n'''
    body     = 'FILE *f = fopen("'+filename+'", "w");\n\nif (!f) exit(1);\nfprintf(f, "%lu\\n", (unsigned long)sizeof('+typeName+'));\n'                 
    self.pushLanguage('CUDA')
    if self.checkRun(includes, body) and os.path.exists(filename):
      f    = file(filename)
      size = int(f.read())
      f.close()
      os.remove(filename)
    else:
      raise RuntimeError('Error checking sizeof(void*) with CUDA')
    if size != self.types.sizes['known-sizeof-void-p']:
      raise RuntimeError('CUDA Error: sizeof(void*) with CUDA compiler is ' + str(size) + ' which differs from sizeof(void*) with C compiler')
    self.popLanguage()

  def configureTypes(self):
    if self.scalartypes.scalartype == 'complex':
      raise RuntimeError('Must use real numbers with CUDA') 
    if not self.scalartypes.precision in ['double', 'single']:
      raise RuntimeError('Must use either single or double precision with CUDA') 
    else:
      self.setCompilers.pushLanguage('CUDA')
      if self.scalartypes.precision == 'double':
        self.setCompilers.addCompilerFlag('-arch sm_13')
      self.setCompilers.popLanguage()
    self.checkSizeofVoidP()
    return

  def checkVersion(self):
    if self.setCompilers.compilerVersionCUDA != self.requiredVersion:
      raise RuntimeError('CUDA Error: PETSc currently requires nvcc version '+self.requiredVersion+' (you have '+self.setCompilers.compilerVersionCUDA+')')
    return

  def configureLibrary(self):
    PETSc.package.NewPackage.configureLibrary(self)
    self.checkVersion()
    if not self.cusp.found or not self.thrust.found:
      raise RuntimeError('PETSc CUDA support requires the CUSP and THRUST packages\nRerun configure using --with-cusp-dir and --with-thrust-dir')
    if self.languages.clanguage == 'C':
      self.addDefine('CUDA_EXTERN_C_BEGIN','extern "C" {')
      self.addDefine('CUDA_EXTERN_C_END','}')
    else:
      self.addDefine('CUDA_EXTERN_C_BEGIN',' ')
      self.addDefine('CUDA_EXTERN_C_END',' ')
    self.configureTypes()
    return
