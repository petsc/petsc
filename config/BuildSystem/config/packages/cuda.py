import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.minversion       = '7.5'
    self.versionname      = 'CUDA_VERSION'
    self.versioninclude   = 'cuda.h'
    self.functions        = ['cublasInit', 'cufftDestroy']
    self.includes         = ['cublas.h','cufft.h','cusparse.h','thrust/version.h']
    self.liblist          = [['libcufft.a', 'libcublas.a','libcudart.a','libcusparse.a'],
                             ['cufft.lib','cublas.lib','cudart.lib','cusparse.lib']]
    self.precisions       = ['single','double']
    self.cxx              = 0
    self.complex          = 1
    self.hastests         = 0
    self.hastestsdatafiles= 0
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers',self)
    self.headers      = framework.require('config.headers',self)
    return

  def getSearchDirectories(self):
    import os
    self.pushLanguage('CUDA')
    petscNvcc = self.getCompiler()
    self.popLanguage()
    self.getExecutable(petscNvcc,getFullPath=1,resultName='systemNvcc')
    if hasattr(self,'systemNvcc'):
      nvccDir = os.path.dirname(self.systemNvcc)
      cudaDir = os.path.split(nvccDir)[0]
      yield cudaDir
    return

  def checkSizeofVoidP(self):
    '''Checks if the CUDA compiler agrees with the C compiler on what size of void * should be'''
    self.log.write('Checking if sizeof(void*) in CUDA is the same as with regular compiler\n')
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
    if 'known-cuda-sizeof-void-p' in self.argDB:
      size = self.argDB['known-cuda-sizeof-void-p']
    elif not self.argDB['with-batch']:
      self.pushLanguage('CUDA')
      if self.checkRun(includes, body) and os.path.exists(filename):
        f    = open(filename)
        size = int(f.read())
        f.close()
        os.remove(filename)
      else:
        raise RuntimeError('Error checking sizeof(void*) with CUDA')
      self.popLanguage()
    else:
      raise RuntimeError('Batch configure does not work with CUDA\nOverride all CUDA configuration with options, such as --known-cuda-sizeof-void-p')
    if size != self.types.sizes['known-sizeof-void-p']:
      raise RuntimeError('CUDA Error: sizeof(void*) with CUDA compiler is ' + str(size) + ' which differs from sizeof(void*) with C compiler')
    self.argDB['known-cuda-sizeof-void-p'] = size
    return

  def configureTypes(self):
    import config.setCompilers
    if not self.getDefaultPrecision() in ['double', 'single']:
      raise RuntimeError('Must use either single or double precision with CUDA')
    self.checkSizeofVoidP()
    return

  def versionToStandardForm(self,ver):
    '''Converts from CUDA 7050 notation to standard notation 7.5'''
    return ".".join(map(str,[int(ver)/1000, int(ver)/10%10]))

  def checkNVCCDoubleAlign(self):
    if 'known-cuda-align-double' in self.argDB:
      if not self.argDB['known-cuda-align-double']:
        raise RuntimeError('CUDA error: PETSC currently requires that CUDA double alignment match the C compiler')
    elif not self.argDB['with-batch']:
      self.pushLanguage('CUDA')
      (outputCUDA,statusCUDA) = self.outputRun('#include <stdio.h>\n','''
        struct {
          double a;
          int    b;
          } teststruct;
        printf("%d",sizeof(teststruct));
        return 0;''')
      self.popLanguage()
      self.pushLanguage('C')
      (outputC,statusC) = self.outputRun('#include <stdio.h>\n','''
        struct {
          double a;
          int    b;
          } teststruct;
        printf("%d",sizeof(teststruct));
        return 0;''')
      self.popLanguage()
      if (statusC or statusCUDA):
        raise RuntimeError('Error compiling check for memory alignment in CUDA')
      if outputC != outputCUDA:
        raise RuntimeError('CUDA compiler error: memory alignment doesn\'t match C compiler (try adding -malign-double to compiler options)')
    else:
      raise RuntimeError('Batch configure does not work with CUDA\nOverride all CUDA configuration with options, such as --known-cuda-align-double')
    return

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    self.checkNVCCDoubleAlign()
    self.configureTypes()
    self.addDefine('HAVE_CUDA','1')
    return
