import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.minversion        = '7.5'
    self.versionname       = 'CUDA_VERSION'
    self.versioninclude    = 'cuda.h'
    self.requiresversion   = 1
    self.functions         = ['cublasInit','cufftDestroy']
    self.includes          = ['cublas.h','cufft.h','cusparse.h','cusolverDn.h','curand.h','thrust/version.h']
    self.basicliblist      = [['libcudart.a'],
                              ['cudart.lib']]
    self.mathliblist       = [['libcufft.a', 'libcublas.a','libcusparse.a','libcusolver.a','libcurand.a'],
                              ['cufft.lib','cublas.lib','cusparse.lib','cusolver.lib','curand.lib']]
    # CUDA provides 2 variants of libcuda.so (for access to CUDA driver API):
    # - fully functional compile, runtime libraries installed with the GPU driver
    #    (for ex:) /usr/lib64/libcuda.so (compile), libcuda.so.1 (runtime)
    # -	stub library - useable only for compiles
    # 	 (for ex:) /usr/local/cuda/lib64/stubs/libcuda.so  (without corresponding libcuda.so.1 for runtime)
    # We are prefering this stub library - as it enables compiles on non-GPU nodes (for ex: login nodes).
    # Using RPATH to this stub location is not appropriate - so skipping via libraries.rpathSkipDirs()
    # Note: PETSc does not use CUDA driver API (as of Sep 29, 2021), but external package for ex: Kokkos does.
    #
    # see more at https://stackoverflow.com/a/52784819
    self.stubliblist       = [['libcuda.so'],
                              ['cuda.lib']]
    self.liblist           = 'dummy' # existence of self.liblist is used by package.py to determine if --with-cuda-lib must be provided
    self.precisions        = ['single','double']
    self.cxx               = 1
    self.complex           = 1
    self.hastests          = 0
    self.hastestsdatafiles = 0
    self.functionsDefine   = ['cusolverDnDpotri']
    self.isnvhpc           = False
    return

  def setupHelp(self, help):
    import nargs
    config.package.Package.setupHelp(self, help)
    help.addArgument('CUDA', '-with-cuda-arch', nargs.ArgString(None, None, 'Cuda architecture for code generation, for example 70, (this may be used by external packages), use all to build a fat binary for distribution'))
    return

  def __str__(self):
    output  = config.package.Package.__str__(self)
    if hasattr(self,'cudaArch'):
      output += '  CUDA SM '+self.cudaArch+'\n'
    if hasattr(self.setCompilers,'CUDA_CXX'):
      output += '  CUDA underlying compiler: CUDA_CXX=' + self.setCompilers.CUDA_CXX + '\n'
    if hasattr(self.setCompilers,'CUDA_CXXFLAGS'):
      output += '  CUDA underlying compiler flags: CUDA_CXXFLAGS=' + self.setCompilers.CUDA_CXXFLAGS + '\n'
    if hasattr(self.setCompilers,'CUDA_CXXLIBS'):
      output += '  CUDA underlying linker libraries: CUDA_CXXLIBS=' + self.setCompilers.CUDA_CXXLIBS + '\n'
    return output

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.scalarTypes  = framework.require('PETSc.options.scalarTypes',self)
    self.compilers    = framework.require('config.compilers',self)
    self.thrust       = framework.require('config.packages.thrust',self)
    self.libraries    = framework.require('config.libraries', self)
    self.odeps        = [self.thrust] # if user supplies thrust, install it first
    return

  def getSearchDirectories(self):
    for i in config.package.Package.getSearchDirectories(self): yield i
    yield self.cudaDir
    return

  def getIncludeDirs(self, prefix, includeDir):
    ''' Generate cuda include dirs'''
    # See comments below in generateLibList() for different prefix formats.
    # format A, prefix = /path/cuda-11.4.0/, includeDir = 'include'. The superclass's method handles this well.
    incDirs = config.package.Package.getIncludeDirs(self, prefix, includeDir)

    if not isinstance(incDirs, list):
      incDirs = [incDirs]

    # format B and C, prefix = /path/nvhpc/Linux_x86_64/21.7/compilers or  /path/nvhpc/Linux_x86_64/21.7/cuda
    nvhpcDir        = os.path.dirname(prefix) # /path/nvhpc/Linux_x86_64/21.7
    nvhpcCudaIncDir = os.path.join(nvhpcDir,'cuda','include')
    nvhpcMathIncDir = os.path.join(nvhpcDir,'math_libs','include')
    if os.path.isdir(nvhpcCudaIncDir) and os.path.isdir(nvhpcMathIncDir):
      incDirs.extend([nvhpcCudaIncDir,nvhpcMathIncDir])

    # format D, prefix = /path/nvhpc/Linux_x86_64/21.7/cuda/11.4
    nvhpcDir           = os.path.dirname(os.path.dirname(prefix))  # /path/nvhpc/Linux_x86_64/21.7
    ver                = os.path.basename(prefix) # 11.4
    nvhpcCudaVerIncDir = os.path.join(nvhpcDir,'cuda',ver,'include')
    nvhpcMathVerIncDir = os.path.join(nvhpcDir,'math_libs',ver,'include')
    if os.path.isdir(nvhpcCudaVerIncDir) and os.path.isdir(nvhpcMathVerIncDir):
      incDirs.extend([nvhpcCudaVerIncDir,nvhpcMathVerIncDir])
    return incDirs

  def generateLibList(self, directory):
    ''' Generate cuda liblist. The difficulty comes from NVIDIA providing two different directory structures through CUDAToolkit and NVHPC'''

    # NVIDIA provides
    # 1) CUDAToolkit, with a directory structure like
    #   /path/cuda-11.4.0/{lib64, lib64/stubs}, here lib64/ contains all basic and math libraries
    #                   +/include
    #                   +/bin/{nvcc,..}
    #
    # 2) NVHPC, with a directory structure like
    # /path/nvhpc/Linux_x86_64/21.7/compilers/bin/{nvcc,nvc,nvc++}
    #                             +/cuda/{include,bin/nvcc,lib64,lib64/stubs}, just symbol links to what in cuda/11.4
    #                             +/cuda/11.4/{include,bin/nvcc,lib64,lib64/stubs}
    #                             +/math_libs/{include,lib64,lib64/stubs}, just symbol links to what in math_libs/11.4
    #                             +/math_libs/11.4/{include,lib64,lib64/stubs}
    #                             +/comm_libs/mpi/bin/{mpicc,mpicxx,mpifort}

    # The input argument 'directory' could be in these formats:
    # A) /path/cuda-11.4.0/lib64,                       by loading a CUDAToolkit or --with-cuda-dir=/path/cuda-11.4.0
    # B) /path/nvhpc/Linux_x86_64/21.7/compilers/lib64, by loading a NVHPC module
    # C) /path/nvhpc/Linux_x86_64/21.7/cuda/lib64,      by --with-cuda-dir=/path/Linux_x86_64/21.7/cuda/
    # D) /path/nvhpc/Linux_x86_64/21.7/cuda/11.4/lib64, by --with-cuda-dir=/path/Linux_x86_64/21.7/cuda/11.4

    # 'directory' is in format A, with basic and math libraries in one directory.
    liblist           = [] # initialize
    toolkitCudaLibDir = directory
    toolkitStubLibDir = os.path.join(toolkitCudaLibDir,'stubs')
    if os.path.isdir(toolkitCudaLibDir) and os.path.isdir(toolkitStubLibDir):
      self.libraries.addRpathSkipDir(toolkitStubLibDir)
      self.liblist = [self.basicliblist[0]+self.mathliblist[0]] + [self.basicliblist[1]+self.mathliblist[1]]
      liblist      = config.package.Package.generateLibList(self, toolkitCudaLibDir)
      self.liblist = self.stubliblist
      stubliblist  = config.package.Package.generateLibList(self,toolkitStubLibDir)
      liblist      = [liblist[0]+stubliblist[0],liblist[1]+stubliblist[1]]

    # 'directory' is in format B or C, and we peel 'directory' two times.
    nvhpcDir        = os.path.dirname(os.path.dirname(directory)) # /path/nvhpc/Linux_x86_64/21.7
    nvhpcCudaLibDir = os.path.join(nvhpcDir,'cuda','lib64')
    nvhpcMathLibDir = os.path.join(nvhpcDir,'math_libs','lib64')
    nvhpcStubLibDir = os.path.join(nvhpcDir,'cuda','lib64','stubs')
    if os.path.isdir(nvhpcCudaLibDir) and os.path.isdir(nvhpcMathLibDir) and os.path.isdir(nvhpcStubLibDir):
      self.libraries.addRpathSkipDir(nvhpcStubLibDir)
      self.liblist = self.basicliblist
      cudaliblist  = config.package.Package.generateLibList(self, nvhpcCudaLibDir)
      self.liblist = self.mathliblist
      mathliblist  = config.package.Package.generateLibList(self, nvhpcMathLibDir)
      self.liblist = self.stubliblist
      stubliblist  = config.package.Package.generateLibList(self, nvhpcStubLibDir)
      liblist.append(mathliblist[0]+cudaliblist[0]+stubliblist[0])
      liblist.append(mathliblist[1]+cudaliblist[1]+stubliblist[1])

    # 'directory' is in format D, and we peel 'directory' three times.
    # We preserve the version info in case a NVHPC installation provides multiple cuda versions and we'd like to respect user's choice
    nvhpcDir           = os.path.dirname(os.path.dirname(os.path.dirname(directory))) # /path/nvhpc/Linux_x86_64/21.7
    ver                = os.path.basename(os.path.dirname(directory)) # 11.4
    nvhpcCudaVerLibDir = os.path.join(nvhpcDir,'cuda',ver,'lib64')
    nvhpcMathVerLibDir = os.path.join(nvhpcDir,'math_libs',ver,'lib64')
    nvhpcStubVerLibDir = os.path.join(nvhpcDir,'cuda',ver,'lib64','stubs')
    if os.path.isdir(nvhpcCudaVerLibDir) and os.path.isdir(nvhpcMathVerLibDir) and os.path.isdir(nvhpcStubVerLibDir):
      self.libraries.addRpathSkipDir(nvhpcStubVerLibDir)
      self.liblist = self.basicliblist
      cudaliblist  = config.package.Package.generateLibList(self, nvhpcCudaVerLibDir)
      self.liblist = self.mathliblist
      mathliblist  = config.package.Package.generateLibList(self, nvhpcMathVerLibDir)
      self.liblist = self.stubliblist
      stubliblist  = config.package.Package.generateLibList(self, nvhpcStubVerLibDir)
      liblist.append(mathliblist[0]+cudaliblist[0]+stubliblist[0])
      liblist.append(mathliblist[1]+cudaliblist[1]+stubliblist[1])
    return liblist

  def checkSizeofVoidP(self):
    '''Checks if the CUDA compiler agrees with the C compiler on what size of void * should be'''
    self.log.write('Checking if sizeof(void*) in CUDA is the same as with regular compiler\n')
    size = self.types.checkSizeof('void *', (8, 4), lang='CUDA', save=False)
    if size != self.types.sizes['void-p']:
      raise RuntimeError('CUDA Error: sizeof(void*) with CUDA compiler is ' + str(size) + ' which differs from sizeof(void*) with C compiler')
    return

  def checkThrustVersion(self,minVer):
    '''Check if thrust version is >= minVer '''
    include = '#include <thrust/version.h> \n#if THRUST_VERSION < ' + str(minVer) + '\n#error "thrust version is too low"\n#endif\n'
    self.pushLanguage('CUDA')
    valid = self.checkCompile(include)
    self.popLanguage()
    return valid

  def configureTypes(self):
    import config.setCompilers
    if not self.getDefaultPrecision() in ['double', 'single']:
      raise RuntimeError('Must use either single or double precision with CUDA')
    self.checkSizeofVoidP()
    # if no user-supplied thrust, check the system's complex ability
    if not self.thrust.found and self.scalarTypes.scalartype == 'complex':
      if self.compilers.cxxDialectRange['CUDA'][1][-2:] < '11':
        raise RuntimeError('CUDA Error: Using CUDA with PetscComplex requires at least c++11 (have {cudadialect}). Run\n$ ./configure --help compilers | grep -A 1 dialect\nfor available options on how to set this'.format(cudadialect=self.compilers.cxxDialect[1]))
      if not self.checkThrustVersion(100908):
        raise RuntimeError('CUDA Error: The thrust library is too low to support PetscComplex. Use --download-thrust or --with-thrust-dir to give a thrust >= 1.9.8')
    return

  def versionToStandardForm(self,ver):
    '''Converts from CUDA 7050 notation to standard notation 7.5'''
    return ".".join(map(str,[int(ver)//1000, int(ver)//10%10]))

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

  def setCudaDir(self):
    import os
    self.pushLanguage('CUDA')
    petscNvcc = self.getCompiler()
    self.popLanguage()
    self.getExecutable(petscNvcc,getFullPath=1,resultName='systemNvcc')
    if hasattr(self,'systemNvcc'):
      self.nvccDir = os.path.dirname(self.systemNvcc) # /path/bin
      d = os.path.dirname(self.nvccDir) # /path
      if os.path.exists(os.path.join(d,'include','cuda.h')): # CUDAToolkit with a structure /path/{bin/nvcc, include/cuda.h}
        self.cudaDir = d
      elif os.path.exists(os.path.normpath(os.path.join(d,'..','cuda','include','cuda.h'))): # NVHPC, see above
        self.cudaDir = os.path.normpath(os.path.join(d,'..','cuda')) # get rid of .. in path, getting /path/Linux_x86_64/21.5/cuda
        self.isnvhpc = True
    else:
      raise RuntimeError('CUDA compiler not found!')
    if not hasattr(self,'cudaDir'):
      raise RuntimeError('CUDA directory not found!')


  def configureLibrary(self):
    import re
    self.setCudaDir()
    # skip this because it does not properly set self.lib and self.include if they have already been set
    if not self.found: config.package.Package.configureLibrary(self)
    self.checkNVCCDoubleAlign()
    self.configureTypes()
    # includes from --download-thrust should override the prepackaged version in cuda - so list thrust.include before cuda.include on the compile command.
    if self.thrust.found:
      self.log.write('Overriding the thrust library in CUDAToolkit with a user-specified one\n')
      self.include = self.thrust.include+self.include

    self.pushLanguage('CUDA')
    petscNvcc = self.getCompiler()
    self.popLanguage()

    genArches = ['30','32', '35', '37', '50', '52', '53', '60','61','70','71', '72', '75', '80']
    if 'with-cuda-arch' in self.framework.clArgDB:
      self.cudaArch = re.search(r'(\d+)$', self.argDB['with-cuda-arch']).group() # get the trailing number from the string
    else:
      dq = os.path.join(self.cudaDir,'extras','demo_suite')
      self.getExecutable('deviceQuery',path = dq)
      if hasattr(self,'deviceQuery'):
        try:
          (out, err, ret) = Configure.executeShellCommand(self.deviceQuery + ' | grep "CUDA Capability"',timeout = 60, log = self.log, threads = 1)
        except Exception as e:
          self.log.write('NVIDIA utility deviceQuery failed '+str(e)+'\n')
        else:
          try:
            out = out.split('\n')[0]
            sm = out[-3:]
            self.cudaArch = str(int(10*float(sm)))
          except:
            self.log.write('Unable to parse the CUDA Capability output from the NVIDIA utility deviceQuery\n')

    if not hasattr(self,'cudaArch') and not self.argDB['with-batch']:
        includes = '''#include <stdio.h>
                    #include <cuda_runtime.h>
                    #include <cuda_runtime_api.h>
                    #include <cuda_device_runtime_api.h>'''
        body = '''int cerr;
                cudaDeviceProp dp;
                cerr = cudaGetDeviceProperties(&dp, 0);
                if (cerr) printf("Error calling cudaGetDeviceProperties\\n");
                else printf("%d\\n",10*dp.major+dp.minor);
                return(cerr);'''
        self.pushLanguage('CUDA')
        try:
          (output,status) = self.outputRun(includes, body)
        except Exception as e:
          self.log.write('petsc-supplied CUDA device query test failed: '+str(e)+'\n')
          self.popLanguage()
        else:
          self.popLanguage()
          self.log.write('petsc-supplied CUDA device query test output: '+output+', status: '+str(status)+'\n')
          if not status:
            try:
              gen = int(output)
            except:
              pass
            else:
              self.log.write('petsc-supplied CUDA device query test found the CUDA Capability is '+str(gen)+'\n')
              self.cudaArch = str(gen)

    if not hasattr(self,'cudaArch'):
      for gen in reversed(genArches):
        self.pushLanguage('CUDA')
        cflags = self.setCompilers.CUDAFLAGS
        self.setCompilers.CUDAFLAGS += ' -gencode arch=compute_'+gen+',code=sm_'+gen
        try:
          valid = self.checkCompile()
        except Exception as e:
          self.log.write('checkCompile on CUDA compile with gencode failed '+str(e)+'\n')
          self.popLanguage()
          self.setCompilers.CUDAFLAGS = cflags
          continue
        else:
          self.popLanguage()
          self.log.write('Flag from checkCompile on CUDA compile with gencode '+str(valid)+'\n')
          if not valid:
            self.setCompilers.CUDAFLAGS = cflags
            continue
          else:
            self.logPrintBox('***** WARNING: Cannot check if gencode '+str(gen)+' works for your hardware, assuming it does.\n\
You may need to run ./configure with-cuda-arch=numerical value (such as 70)\n\
to set the right generation for your hardware.')
            self.cudaArch = gen
            self.setCompilers.CUDAFLAGS = cflags
            break

    if hasattr(self,'cudaArch'):
      if self.cudaArch == 'all':
        for gen in genArches:
          self.setCompilers.CUDAFLAGS += ' -gencode arch=compute_'+gen+',code=sm_'+gen+' '
          self.log.write(self.setCompilers.CUDAFLAGS+'\n')
        self.addDefine('CUDA_GENERATION','0')
      else:
        self.setCompilers.CUDAFLAGS += ' -gencode arch=compute_'+self.cudaArch+',code=sm_'+self.cudaArch+' '
        self.addDefine('CUDA_GENERATION',self.cudaArch)

    self.addDefine('HAVE_CUDA','1')
    if not self.version_tuple:
      self.checkVersion(); # set version_tuple
    if self.version_tuple[0] >= 11:
      self.addDefine('HAVE_CUDA_VERSION_11PLUS','1')

    # determine the compiler used by nvcc
    (out, err, ret) = Configure.executeShellCommand(petscNvcc + ' ' + self.setCompilers.CUDAFLAGS + ' --dryrun dummy.cu 2>&1 | grep D__CUDACC__ | head -1 | cut -f2 -d" "')
    if out:
      # MPI.py adds its include paths and libraries to these lists and saves them again
      self.setCompilers.CUDA_CXX = out
      self.setCompilers.CUDA_CXXFLAGS = ''
      self.setCompilers.CUDA_CXXLIBS = ''
      self.logPrint('Determined the compiler nvcc uses is ' + out);
      self.logPrint('PETSc C compiler '+self.compilers.CC)
      self.logPrint('PETSc C++ compiler '+self.compilers.CXX)

      # TODO: How to handle MPI compiler wrapper as opposed to its underlying compiler
      if out == self.compilers.CC or out == self.compilers.CXX:
        # nvcc will say it is using gcc as its compiler, it pass a flag when using to
        # treat it as a C++ compiler
        newFlags = self.setCompilers.CPPFLAGS.split()+self.setCompilers.CFLAGS.split()+self.setCompilers.CXXPPFLAGS.split()+self.setCompilers.CXXFLAGS.split()
        # need to remove the std flag from the list, nvcc will already have its own flag set
        # With IBM XL compilers, we also need to remove -+
        self.setCompilers.CUDA_CXXFLAGS = ' '.join([flg for flg in newFlags if not flg.startswith(('-std=c++','-std=gnu++','-+'))])
      else:
        # only add any -I arguments since compiler arguments may not work
        flags = self.setCompilers.CPPFLAGS.split(' ')+self.setCompilers.CFLAGS.split(' ')+self.setCompilers.CXXFLAGS.split(' ')
        for i in flags:
          if i.startswith('-I'):
            self.setCompilers.CUDA_CXXFLAGS += ' '+i
      # set compiler flags for compiler called by nvcc
      if self.setCompilers.CUDA_CXXFLAGS:
        self.addMakeMacro('CUDA_CXXFLAGS',self.setCompilers.CUDA_CXXFLAGS)
      else:
        self.logPrint('No CUDA_CXXFLAGS available')
      self.addMakeMacro('CUDA_CXX',self.setCompilers.CUDA_CXX)

      # Intel compiler environment breaks GNU compilers, fix it just enough to allow g++ to run
      if self.setCompilers.CUDA_CXX == 'gcc' and config.setCompilers.Configure.isIntel(self.compilers.CXX,self.log):
        self.logPrint('''Removing Intel's CPLUS_INCLUDE_PATH when using nvcc since it breaks g++''')
        self.delMakeMacro('CUDAC')
        self.addMakeMacro('CUDAC','CPLUS_INCLUDE_PATH="" '+petscNvcc)
    else:
      self.logPrint('nvcc --dryrun failed, unable to determine CUDA_CXX and CUDA_CXXFLAGS')
    return
