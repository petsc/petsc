import PETSc.package
import os

class Configure(PETSc.package.NewPackage):
  def __init__(self, framework):
    PETSc.package.NewPackage.__init__(self, framework)
    self.functions        = ['cublasInit', 'cufftDestroy']
    self.includes         = ['cublas.h', 'cufft.h','cusparse.h']
    self.liblist          = [['libcufft.a', 'libcublas.a','libcudart.a','libcusparse.a'],
                             ['cufft.lib','cublas.lib','cudart.lib','cusparse.lib']]
    self.double           = 0   # 1 means requires double precision 
    self.cxx              = 0
    self.requires32bitint = 0
    self.worksonWindows   = 1

    self.cudaArch      = ''
    self.CUDAVersion   = '4000' # Version 4.0
    self.CUSPVersion   = '200' #Version 0.2.0
    self.ThrustVersion = '100400' #Version 1.4.0
#
#   obtain thrust and cusp with
#   hg clone https://thrust.googlecode.com/hg/ thrust 
#   hg clone https://cusp-library.googlecode.com/hg/ cusp
#     put them in /usr/local/cuda
#

    # Get Thrust from hg clone https://thrust.googlecode.com/hg/ thrust
    # Get CUSP from hg clone https://cusp-library.googlecode.com/hg/

    self.CUDAVersionStr = str(int(self.CUDAVersion)/1000) + '.' + str(int(self.CUDAVersion)%100)
    self.ThrustVersionStr = str(int(self.ThrustVersion)/100000) + '.' + str(int(self.ThrustVersion)/100%1000) + '.' + str(int(self.ThrustVersion)%100)
    self.CUSPVersionStr   = str(int(self.CUSPVersion)/100000) + '.' + str(int(self.CUSPVersion)/100%1000) + '.' + str(int(self.CUSPVersion)%100)
    return

  def __str__(self):
    output  = PETSc.package.NewPackage.__str__(self)
    output += '  Arch:     '+self.cudaArch+'\n'
    return output

  def setupHelp(self, help):
    import nargs
    PETSc.package.NewPackage.setupHelp(self, help)
    help.addArgument('CUDA', '-with-cuda-arch=<arch>', nargs.Arg(None, None, 'Target architecture for nvcc, e.g. sm_13'))
    help.addArgument('CUDA', '-with-cuda-only=<bool>', nargs.ArgBool(None, 1, 'Allows CUDA compiles without Thrust and Cusp'))
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
    self.libdir           = os.path.join('lib','Win32')
    self.altlibdir        = os.path.join('lib','x64')
    yield(os.path.join('/cygdrive','c','Program Files','NVIDIA GPU Computing Toolkit','CUDA','v4.0'))
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
    if self.scalartypes.scalartype == 'complex':
      raise RuntimeError('Must use real numbers with CUDA') 
    if not config.setCompilers.Configure.isGNU(self.setCompilers.CC):
      raise RuntimeError('Must use GNU compilers with CUDA')
    if not self.scalartypes.precision in ['double', 'single']:
      raise RuntimeError('Must use either single or double precision with CUDA') 
    else:
      self.setCompilers.pushLanguage('CUDA')
#Not setting -arch if with-cuda-arch is not specified uses nvcc default architecture
      if 'with-cuda-arch' in self.framework.argDB:
        if not self.framework.argDB['with-cuda-arch'] in ['compute_10', 'compute_11', 'compute_12', 'compute_13', 'compute_20', 'sm_10', 'sm_11', 'sm_12', 'sm_13', 'sm_20', 'sm_21']:
          raise RuntimeError('CUDA Error: specified CUDA architecture invalid.  Example of valid architecture: \'-with-cuda-arch=sm_13\'')
        else:
          self.cudaArch = '-arch='+ self.framework.argDB['with-cuda-arch']
      elif self.scalartypes.precision == 'double':
        #default to sm_13 for double precision
        self.cudaArch = '-arch=sm_13'
      elif self.scalartypes.precision == 'single':
        #default to sm_10 for single precision
        self.cudaArch = '-arch=sm_10'
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
      self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.thrust.include)
      if not self.checkRun('#include <cuda.h>\n#include <stdio.h>', 'if (CUDA_VERSION < ' + self.CUDAVersion +') {printf("Invalid version %d\\n", CUDA_VERSION); return 1;}'):
        raise RuntimeError('CUDA version error: PETSC currently requires CUDA version '+self.CUDAVersionStr+' or higher - when compiling with CUDA')
      self.compilers.CUDAPPFLAGS = oldFlags
      self.popLanguage()
    else:
      raise RuntimeError('Batch configure does not work with CUDA\nOverride all CUDA configuration with options, such as --known-cuda-version')
    return

  def checkThrustVersion(self):
    if 'known-thrust-version' in self.argDB:
      if self.argDB['known-thrust-version'] < self.ThrustVersion:
        raise RuntimeError('Thrust version error '+self.argDB['known-thrust-version']+' < '+self.ThrustVersion+': PETSC currently requires Thrust version '+self.ThrustVersionStr+' or higher when compiling with CUDA')
    elif not self.argDB['with-batch']:
      self.pushLanguage('CUDA')
      oldFlags = self.compilers.CUDAPPFLAGS
      self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.thrust.include)
      if not self.checkRun('#include <thrust/version.h>\n#include <stdio.h>', 'if (THRUST_VERSION < ' + self.ThrustVersion +') {printf("Invalid version %d\\n", THRUST_VERSION); return 1;}'):
        raise RuntimeError('Thrust version error: PETSC currently requires Thrust version '+self.ThrustVersionStr+' or higher - when compiling with CUDA')
      self.compilers.CUDAPPFLAGS = oldFlags
      self.popLanguage()
    else:
      raise RuntimeError('Batch configure does not work with CUDA\nOverride all CUDA configuration with options, such as --known-thrust-version')
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

  def checkCUSPVersion(self):
    if 'known-cusp-version' in self.argDB:
      if self.argDB['known-cusp-version'] < self.CUSPVersion:
        raise RuntimeError('CUSP version error '+self.argDB['known-cusp-version']+' < '+self.CUSPVersion+': PETSC currently requires CUSP version '+self.CUSPVersionStr+' or higher when compiling with CUDA')
    elif not self.argDB['with-batch']:
      self.pushLanguage('CUDA')
      oldFlags = self.compilers.CUDAPPFLAGS
      self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.cusp.include)
      self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.thrust.include)
      if not self.checkRun('#include <cusp/version.h>\n#include <stdio.h>', 'if (CUSP_VERSION < ' + self.CUSPVersion +') {printf("Invalid version %d\\n", CUSP_VERSION); return 1;}'):
        raise RuntimeError('CUSP version error: PETSC currently requires CUSP version '+self.CUSPVersionStr+' or higher - when compiling with CUDA')
      self.compilers.CUDAPPFLAGS = oldFlags
      self.popLanguage()
    else:
      raise RuntimeError('Batch configure does not work with CUDA\nOverride all CUDA configuration with options, such as --known-cusp-version')
    return

  def configureLibrary(self):
    PETSc.package.NewPackage.configureLibrary(self)
    self.checkCUDAVersion()
    self.checkNVCCDoubleAlign()
    if not self.framework.argDB['with-cuda-only']:
      if not self.cusp.found or not self.thrust.found:
        raise RuntimeError('PETSc CUDA support requires the CUSP and Thrust packages\nRerun configure using --with-cusp-dir and --with-thrust-dir')
      self.checkThrustVersion()
      self.checkCUSPVersion()
    if self.languages.clanguage == 'C':
      self.addDefine('CUDA_EXTERN_C_BEGIN','extern "C" {')
      self.addDefine('CUDA_EXTERN_C_END','}')
    else:
      self.addDefine('CUDA_EXTERN_C_BEGIN',' ')
      self.addDefine('CUDA_EXTERN_C_END',' ')
    self.configureTypes()
    return
