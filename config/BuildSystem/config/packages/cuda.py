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
    self.basicliblist      = [['libcuda.a','libcudart.a'],
                              ['cuda.lib','cudart.lib'],
                              ['libcudart.a'],
                              ['cudart.lib']]
    self.mathliblist       = [['libcufft.a', 'libcublas.a','libcusparse.a','libcusolver.a','libcurand.a'],
                              ['cufft.lib','cublas.lib','cusparse.lib','cusolver.lib','curand.lib']]
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
    self.odeps        = [self.thrust] # if user supplies thrust, install it first
    return

  def getSearchDirectories(self):
    for i in config.package.Package.getSearchDirectories(self): yield i
    yield self.cudaDir
    return

  def getIncludeDirs(self, prefix, includeDir):
    incDirs = config.package.Package.getIncludeDirs(self, prefix, includeDir)
    nvhpcDir        = os.path.dirname(prefix) # /path/Linux_x86_64/21.5
    nvhpcCudaIncDir = os.path.join(nvhpcDir,'cuda','include')
    nvhpcMathIncDir = os.path.join(nvhpcDir,'math_libs','include')
    if os.path.isdir(nvhpcCudaIncDir) and os.path.isdir(nvhpcMathIncDir):
      if isinstance(incDirs, list):
        return incDirs.extend([nvhpcCudaIncDir,nvhpcMathIncDir])
      else:
        return [incDirs,nvhpcCudaIncDir,nvhpcMathIncDir]
    return incDirs

  def generateLibList(self, directory):
    '''NVHPC separated the libraries into a different math_libs directory and the directory with the basic CUDA library'''
    '''Thus configure needs to support finding both sets of libraries and include files given a single directory that points to CUDA directory'''
    '''Note the difficulty comes from the fact that math libraries are ABOVE the CUDA version in the directory tree'''

    verdir = os.path.dirname(directory)
    ver = os.path.basename(verdir)
    mdir = os.path.join('..','..','math_libs',ver)
    if os.path.isdir(os.path.join(verdir,mdir,'include')):
      self.includedir = [os.path.join(mdir,'include'), 'include']

    # first try the standard list with all libraries in one directory
    self.liblist =  [self.basicliblist[0]+self.mathliblist[0]]+[self.basicliblist[1]+self.mathliblist[1]]
    self.liblist += [self.basicliblist[2]+self.mathliblist[0]]+[self.basicliblist[3]+self.mathliblist[1]]
    liblist = config.package.Package.generateLibList(self, directory)

    # create list with math libraries separate
    lib = os.path.basename(directory)
    ver = os.path.basename(os.path.dirname(directory))
    newdirectory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(directory))),'math_libs',ver,lib)
    if os.path.isdir(newdirectory):
      self.liblist = [self.basicliblist[0]]
      subliblist = config.package.Package.generateLibList(self, directory)
      self.liblist = [self.mathliblist[0]]
      mathsubliblist = config.package.Package.generateLibList(self, newdirectory)
      liblist = [liblist[0],liblist[1],mathsubliblist[0] + subliblist[0]]

    # When 'directory' is in format like /path/Linux_x86_64/21.5/compilers/lib, NVHPC directory structure is like
    # /path/Linux_x86_64/21.5/compilers/bin/{nvcc,nvc,nvc++}
    #                       +/comm_libs/mpi/bin/{mpicc,mpicxx,mpifort}
    #                       +/cuda/{include,lib64}
    #                       +/math_libs/{include,lib64}
    nvhpcDir        = os.path.dirname(os.path.dirname(directory)) # /path/Linux_x86_64/21.5
    nvhpcCudaLibDir = os.path.join(nvhpcDir,'cuda','lib64')
    nvhpcMathLibDir = os.path.join(nvhpcDir,'math_libs','lib64')
    if os.path.isdir(nvhpcCudaLibDir) and os.path.isdir(nvhpcMathLibDir):
      self.liblist    = [self.basicliblist[0]]
      subliblist      = config.package.Package.generateLibList(self, nvhpcCudaLibDir)
      self.liblist    = [self.mathliblist[0]]
      mathsubliblist  = config.package.Package.generateLibList(self, nvhpcMathLibDir)
      liblist = [liblist[0],liblist[1],mathsubliblist[0] + subliblist[0]]
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
    if not self.thrust.found and self.scalarTypes.scalartype == 'complex': # if no user-supplied thrust, check the system's complex ability
      if not self.compilers.cxxdialect in ['C++11','C++14'] or not self.compilers.cudadialect in ['C++11','C++14']:
        raise RuntimeError('CUDA Error: Using CUDA with PetscComplex requires a C++ dialect at least cxx11. Use --with-cxx-dialect=xxx and --with-cuda-dialect=xxx to specify a suitable compiler')
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
      self.nvccDir = os.path.dirname(self.systemNvcc)
      d = os.path.split(self.nvccDir)[0]
      if os.path.exists(os.path.join(d,'include','cuda.h')):
        self.cudaDir = d
      elif os.path.exists(os.path.join(d,'..','cuda','include','cuda.h')):
        self.cudaDir = os.path.join(d,'..','cuda')
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
