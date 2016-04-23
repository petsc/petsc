import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.functions        = ['cublasInit', 'cufftDestroy']
    self.includes         = ['cublas.h','cufft.h','cusparse.h','thrust/version.h']
    self.liblist          = [['libcufft.a', 'libcublas.a','libcudart.a','libcusparse.a'],
                             ['cufft.lib','cublas.lib','cudart.lib','cusparse.lib']]
    self.double           = 0   # 1 means requires double precision
    self.cxx              = 0
    self.complex          = 1
    self.cudaArch         = ''
    self.CUDAVersion      = '4200' # Minimal cuda version is 4.2
    self.CUDAVersionStr   = str(int(self.CUDAVersion)/1000) + '.' + str(int(self.CUDAVersion)/100%10)
    self.hastests         = 1
    self.hastestsdatafiles= 1
    return

  def __str__(self):
    output  = config.package.Package.__str__(self)
    output += '  Arch:     '+self.cudaArch+'\n'
    return output

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('CUDA', '-with-cuda-arch=<arch>', nargs.Arg(None, None, 'Target architecture for nvcc, e.g. sm_20'))
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers',self)
    self.headers      = framework.require('config.headers',self)
    self.scalartypes  = framework.require('PETSc.options.scalarTypes', self)
    self.languages    = framework.require('PETSc.options.languages',   self)
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
#    if self.scalartypes.scalartype == 'complex':
#      raise RuntimeError('Must use real numbers with CUDA')
    if not config.setCompilers.Configure.isGNU(self.setCompilers.CC, self.log):
      raise RuntimeError('Must use GNU compilers with CUDA')
    if not self.scalartypes.precision in ['double', 'single']:
      raise RuntimeError('Must use either single or double precision with CUDA')
    else:
      self.setCompilers.pushLanguage('CUDA')
#Not setting -arch if with-cuda-arch is not specified uses nvcc default architecture
      if 'with-cuda-arch' in self.argDB:
        if not self.argDB['with-cuda-arch'] in ['compute_10', 'compute_11', 'compute_12', 'compute_13', 'compute_20', 'compute_21', 'compute_30', 'compute_35', 'compute_50', 'sm_10', 'sm_11', 'sm_12', 'sm_13', 'sm_20', 'sm_21', 'sm_30', 'sm_35', 'sm_50']:
          raise RuntimeError('CUDA Error: specified CUDA architecture invalid.  Example of valid architecture: \'-with-cuda-arch=sm_20\'')
        else:
          self.cudaArch = '-arch='+ self.argDB['with-cuda-arch']
      else :
        # default to sm_20 because cuda 6.5 emits deprecation warning for
        # earlier architectures
        self.cudaArch = '-arch=sm_20'
      if self.cudaArch:
        self.setCompilers.addCompilerFlag(self.cudaArch)
      self.setCompilers.popLanguage()
    self.checkSizeofVoidP()
    return

  def checkCUDAVersion(self):
    if 'known-cuda-version' in self.argDB:
      if self.argDB['known-cuda-version'] < self.CUDAVersion:
        raise RuntimeError('CUDA version error '+self.argDB['known-cuda-version']+' < '+self.CUDAVersion+': PETSC currently requires CUDA version '+self.CUDAVersionStr+' or higher when compiling with CUDA')
    elif not self.argDB['with-batch']:
      self.pushLanguage('CUDA')
      oldFlags = self.compilers.CUDAPPFLAGS
      if not self.checkRun('#include <cuda.h>\n#include <stdio.h>', 'if (CUDA_VERSION < ' + self.CUDAVersion +') {printf("Invalid version %d\\n", CUDA_VERSION); return 1;}'):
        raise RuntimeError('CUDA version error: PETSC currently requires CUDA version '+self.CUDAVersionStr+' or higher - when compiling with CUDA')
      self.compilers.CUDAPPFLAGS = oldFlags
      self.popLanguage()
    else:
      raise RuntimeError('Batch configure does not work with CUDA\nOverride all CUDA configuration with options, such as --known-cuda-version')
    if self.defaultScalarType.lower() == 'complex':
      CUDAVersionComplex    = '7000' # Minimal cuda version for complex is 7.0
      CUDAVersionComplexStr = str(int(CUDAVersionComplex)/1000) + '.' + str(int(CUDAVersionComplex)/100%10)
      if 'known-cuda-version' in self.argDB:
        if self.argDB['known-cuda-version'] < CUDAVersionComplex:
          raise RuntimeError('CUDA version error '+self.argDB['known-cuda-version']+' < '+CUDAVersionComplex+': PETSC currently requires CUDA version '+CUDAVersionComplexStr+' or higher when compiling with CUDA')
      elif not self.argDB['with-batch']:
        self.pushLanguage('CUDA')
        oldFlags = self.compilers.CUDAPPFLAGS
        if not self.checkRun('#include <cuda.h>\n#include <stdio.h>', 'if (CUDA_VERSION < ' + CUDAVersionComplex +') {printf("Invalid version %d\\n", CUDA_VERSION); return 1;}'):
          raise RuntimeError('CUDA version error: PETSC currently requires CUDA version '+CUDAVersionComplexStr+' or higher - when compiling with CUDA and complex scalars')
        self.compilers.CUDAPPFLAGS = oldFlags
        self.popLanguage()
      else:
        raise RuntimeError('Batch configure does not work with CUDA\nOverride all CUDA configuration with options, such as --known-cuda-version')
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
    raise RuntimeError('Please use master branch from petsc git clone for CUDA functionality')
    config.package.Package.configureLibrary(self)
    self.checkCUDAVersion()
    self.checkNVCCDoubleAlign()
    self.configureTypes()
    return
