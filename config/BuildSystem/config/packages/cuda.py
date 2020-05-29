import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.minversion        = '7.5'
    self.versionname       = 'CUDA_VERSION'
    self.versioninclude    = 'cuda.h'
    self.requiresversion   = 1
    self.functions         = ['cublasInit', 'cufftDestroy','cuInit']
    self.includes          = ['cublas.h','cufft.h','cusparse.h','cusolverDn.h','thrust/version.h']
    self.liblist           = [['libcufft.a', 'libcublas.a','libcudart.a','libcusparse.a','libcusolver.a','libcuda.a'],
                              ['cufft.lib','cublas.lib','cudart.lib','cusparse.lib','cusolver.lib','cuda.lib']]
    self.precisions        = ['single','double']
    self.cxx               = 0
    self.complex           = 1
    self.hastests          = 0
    self.hastestsdatafiles = 0
    self.gencodearch       = ''
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('CUDA', '-with-cuda-gencodearch', nargs.ArgInt(None, 0, 'Cuda architecture for code generation (may be used by external packages)'))
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
      self.nvccDir = os.path.dirname(self.systemNvcc)
      self.cudaDir = os.path.split(self.nvccDir)[0]
      yield self.cudaDir
    return

  def checkSizeofVoidP(self):
    '''Checks if the CUDA compiler agrees with the C compiler on what size of void * should be'''
    self.log.write('Checking if sizeof(void*) in CUDA is the same as with regular compiler\n')
    size = self.types.checkSizeof('void *', (8, 4), lang='CUDA', save=False)
    if size != self.types.sizes['void-p']:
      raise RuntimeError('CUDA Error: sizeof(void*) with CUDA compiler is ' + str(size) + ' which differs from sizeof(void*) with C compiler')
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
    else:
      typedef = 'typedef struct {double a; int b;} teststruct;\n'
      cuda_size = self.types.checkSizeof('teststruct', (16, 12), lang='CUDA', codeBegin=typedef, save=False)
      c_size = self.types.checkSizeof('teststruct', (16, 12), lang='C', codeBegin=typedef, save=False)
      if c_size != cuda_size:
        raise RuntimeError('CUDA compiler error: memory alignment doesn\'t match C compiler (try adding -malign-double to compiler options)')
    return

  def configureLibrary(self):
    config.package.Package.configureLibrary(self)
    self.checkNVCCDoubleAlign()
    self.configureTypes()
    gencodearch = self.argDB['with-cuda-gencodearch']
    if gencodearch:
      self.gencodearch = str(gencodearch)
    self.addDefine('HAVE_CUDA','1')
    return
