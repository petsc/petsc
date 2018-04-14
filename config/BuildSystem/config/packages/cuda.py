import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions        = ['cublasInit', 'cufftDestroy']
    self.includes         = ['cublas.h','cufft.h','cusparse.h','thrust/version.h']
    self.liblist          = [['libcufft.a', 'libcublas.a','libcudart.a','libcusparse.a'],
                             ['cufft.lib','cublas.lib','cudart.lib','cusparse.lib']]
    self.precisions       = ['single','double']
    self.cxx              = 0
    self.complex          = 1
    self.cudaArch         = ''
    self.CUDAVersion      = ''
    self.CUDAMinVersion   = '7050' # Minimal cuda version is 7.5
    self.hastests         = 0
    self.hastestsdatafiles= 0
    return

  def __str__(self):
    output  = config.package.Package.__str__(self)
    output += '  Arch:     '+self.cudaArch+'\n'
    return output

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
        f    = file(filename)
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
    if not config.setCompilers.Configure.isGNU(self.setCompilers.CC, self.log):
      raise RuntimeError('Must use GNU compilers with CUDA')
    if not self.getDefaultPrecision() in ['double', 'single']:
      raise RuntimeError('Must use either single or double precision with CUDA')
    self.checkSizeofVoidP()
    return

  def verToStr(self,ver):
    return str(int(ver)/1000) + '.' + str(int(ver)/10%10)

  def checkCUDAVersion(self):
    import re
    HASHLINESPACE = ' *(?:\n#.*\n *)*'
    self.pushLanguage('CUDA')
    oldFlags = self.compilers.CUDAPPFLAGS
    self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.include)
    cuda_test = '#include <cuda.h>\nint cuda_ver = CUDA_VERSION;\n'
    if self.checkCompile(cuda_test):
      buf = self.outputPreprocess(cuda_test)
      try:
        self.CUDAVersion = re.compile('\nint cuda_ver ='+HASHLINESPACE+'([0-9]+)'+HASHLINESPACE+';').search(buf).group(1)
      except:
        self.logPrint('Unable to parse CUDA version from header. Probably a buggy preprocessor')
    self.compilers.CUDAPPFLAGS = oldFlags
    self.popLanguage()
    if self.CUDAVersion and self.CUDAVersion < self.CUDAMinVersion:
      raise RuntimeError('CUDA version error: PETSC currently requires CUDA version '+self.verToStr(self.CUDAMinVersion)+' or higher. Found version '+self.verToStr(self.CUDAVersion))
    return

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
    self.checkCUDAVersion()
    self.checkNVCCDoubleAlign()
    self.configureTypes()
    self.addDefine('HAVE_VECCUDA','1')
    return
