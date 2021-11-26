import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)

    self.minversion       = '3.8'
    self.versionname      = 'HIP_VERSION_MAJOR.HIP_VERSION_MINOR'
    self.versioninclude   = 'hip/hip_version.h'
    self.requiresversion  = 1
    self.functionsCxx     = [1,'', 'hipblasCreate']
    self.includes         = ['hipblas.h','hipsparse.h']
    self.liblist          = [['libhipsparse.a','libhipblas.a','librocsparse.a','librocsolver.a','librocblas.a','librocrand.a','libamdhip64.a'],
                             ['hipsparse.lib','hipblas.lib','rocsparse.lib','rocsolver.lib','rocblas.lib','rocrand.lib','amdhip64.lib'],]
    self.precisions       = ['single','double']
    self.buildLanguages   = ['HIP']
    self.complex          = 1
    self.hastests         = 0
    self.hastestsdatafiles= 0
    self.devicePackage    = 1
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('HIP', '-with-hip-arch', nargs.ArgString(None, None, 'AMD GPU architecture for code generation, for example gfx908, (this may be used by external packages)'))
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers',self)
    self.headers      = framework.require('config.headers',self)
    return

  def __str__(self):
    output  = config.package.Package.__str__(self)
    if hasattr(self,'hipArch'):
      output += '  HIP arch: '+ self.hipArch +'\n'
    return output

  def getSearchDirectories(self):
    import os
    self.pushLanguage('HIP')
    petscHip = self.getCompiler()
    self.popLanguage()
    self.getExecutable(petscHip,getFullPath=1,resultName='systemHipc')
    if hasattr(self,'systemHipc'):
      hipcDir = os.path.dirname(self.systemHipc)
      hipDir = os.path.split(hipcDir)[0]
      yield hipDir
    return

  def checkSizeofVoidP(self):
    '''Checks if the HIPC compiler agrees with the C compiler on what size of void * should be'''
    self.log.write('Checking if sizeof(void*) in HIP is the same as with regular compiler\n')
    size = self.types.checkSizeof('void *', (8, 4), lang='HIP', save=False)
    if size != self.types.sizes['void-p']:
      raise RuntimeError('HIP Error: sizeof(void*) with HIP compiler is ' + str(size) + ' which differs from sizeof(void*) with C compiler')
    return

  def configureTypes(self):
    import config.setCompilers
    if not self.getDefaultPrecision() in ['double', 'single']:
      raise RuntimeError('Must use either single or double precision with HIP')
    self.checkSizeofVoidP()
    return

  def checkHIPCDoubleAlign(self):
    if 'known-hip-align-double' in self.argDB:
      if not self.argDB['known-hip-align-double']:
        raise RuntimeError('HIP error: PETSC currently requires that HIP double alignment match the C compiler')
    else:
      typedef = 'typedef struct {double a; int b;} teststruct;\n'
      hip_size = self.types.checkSizeof('teststruct', (16, 12), lang='HIP', codeBegin=typedef, save=False)
      c_size = self.types.checkSizeof('teststruct', (16, 12), lang='C', codeBegin=typedef, save=False)
      if c_size != hip_size:
        raise RuntimeError('HIP compiler error: memory alignment doesn\'t match C compiler (try adding -malign-double to compiler options)')
    return

  def configureLibrary(self):
    self.getExecutable('hipconfig',getFullPath=1,resultName='hip_config')
    if hasattr(self,'hip_config'):
      try:
        self.platform = config.package.Package.executeShellCommand([self.hip_config,'--platform'],log=self.log)[0]
      except RuntimeError:
        pass

    # Handle the platform issues
    if not hasattr(self,'platform'):
      if 'HIP_PLATFORM' in os.environ:
        self.platform = os.environ['HIP_PLATFORM']
      elif hasattr(self,'systemNvcc'):
        self.platform = 'nvidia'
      else:
        self.platform = 'amd'

    self.libraries.pushLanguage('HIP')
    self.addDefine('HAVE_HIP','1')
    if self.platform in ['nvcc','nvidia']:
      self.pushLanguage('CUDA')
      petscNvcc = self.getCompiler()
      cudaFlags = self.getCompilerFlags()
      self.popLanguage()
      self.getExecutable(petscNvcc,getFullPath=1,resultName='systemNvcc')
      if hasattr(self,'systemNvcc'):
        nvccDir = os.path.dirname(self.systemNvcc)
        cudaDir = os.path.split(nvccDir)[0]
      else:
        raise RuntimeError('Unable to locate CUDA NVCC compiler')
      self.includedir = ['include',os.path.join(cudaDir,'include')]
      self.delDefine('HAVE_CUDA')
      self.addDefine('HAVE_HIPCUDA',1)
      self.framework.addDefine('__HIP_PLATFORM_NVCC__',1) # deprecated from 4.3.0
      self.framework.addDefine('__HIP_PLATFORM_NVIDIA__',1)
    else:
      self.addDefine('HAVE_HIPROCM',1)
      self.framework.addDefine('__HIP_PLATFORM_HCC__',1) # deprecated from 4.3.0
      self.framework.addDefine('__HIP_PLATFORM_AMD__',1)
      if 'with-hip-arch' in self.framework.clArgDB:
        self.hipArch = self.argDB['with-hip-arch']
      else:
        self.getExecutable('rocminfo',getFullPath=1)
        if hasattr(self,'rocminfo'):
          try:
            (out, err, ret) = Configure.executeShellCommand(self.rocminfo + ' | grep " gfx" ',timeout = 60, log = self.log, threads = 1)
          except Exception as e:
            self.log.write('ROCM utility ' + self.rocminfo + ' failed: '+str(e)+'\n')
          else:
            try:
              s = set([i for i in out.split() if 'gfx' in i])
              self.hipArch = list(s)[0]
              self.log.write('ROCM utility ' + self.rocminfo + ' said the HIP arch is ' + self.hipArch + '\n')
            except:
              self.log.write('Unable to parse the ROCM utility ' + self.rocminfo + '\n')
      if hasattr(self,'hipArch'):
        self.hipArch.lower() # to have a uniform format even if user set hip arch in weird cases
        if not self.hipArch.startswith('gfx'):
          raise RuntimeError('HIP arch name ' + self.hipArch + ' is not in the supported gfxnnn format')
        self.setCompilers.HIPFLAGS += ' --amdgpu-target=' + self.hipArch +' '
      else:
        raise RuntimeError('You must set --with-hip-arch=gfx900, gfx906, gfx908, gfx90a etc or make ROCM utility "rocminfo" available on your PATH')

    config.package.Package.configureLibrary(self)
    #self.checkHIPDoubleAlign()
    self.configureTypes()
    self.libraries.popLanguage()
    return
