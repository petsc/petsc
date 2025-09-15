import config.package
import os

# Dev note: set the Kokkos env var KOKKOS_DISABLE_WARNINGS=1 at runtime to disable warnings in Kokkos initialization

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = '4.7.01'
    self.minversion       = '3.7.01'
    self.versionname      = 'KOKKOS_VERSION'
    self.download         = ['git://https://github.com/kokkos/kokkos.git','https://github.com/kokkos/kokkos/archive/'+self.gitcommit+'.tar.gz']
    self.downloaddirnames = ['kokkos']
    self.excludedDirs     = ['kokkos-kernels'] # Do not wrongly think kokkos-kernels as kokkos-vernum
    self.includes         = ['Kokkos_Macros.hpp']
    self.liblist          = [['libkokkoscontainers.a','libkokkoscore.a','libkokkossimd.a'],
                             ['libkokkoscontainers.a','libkokkoscore.a']]
    self.functions        = ['']
    self.functionsCxx     = [1,'namespace Kokkos {void initialize(int&,char*[]);}','int one = 1;char* args[1];Kokkos::initialize(one,args);']
    self.minCxxVersion    = 'c++17'
    # nvcc_wrapper in Kokkos-4.0.00 does not handle -std=c++20 correctly (it wrongly passes that to -Xcompiler).
    # Though Kokkos/develop fixed the problem, we set maxCxxVersion to c++17 here to lower the standard PETSc would use as a workaround.
    # TODO: remove this line once we use newer Kokkos versions
    self.maxCxxVersion    = 'c++17'
    self.buildLanguages   = ['Cxx'] # Depending on if cuda, hip or sycl is available, it will be modified.
    self.hastests         = 1
    self.requiresrpath    = 1
    self.precisions       = ['single','double']
    self.devicePackage    = 1 # we treat Kokkos as a device package, though it might run without GPUs.
    self.minCmakeVersion  = (3,16,0)
    self.macros           = ['KOKKOS_ENABLE_THREADS', 'KOKKOS_ENABLE_CUDA', 'KOKKOS_ENABLE_HIP', 'KOKKOS_ENABLE_SYCL']
    return

  def __str__(self):
    output  = config.package.CMakePackage.__str__(self)
    if hasattr(self,'system'): output += '  Backends: '+str(self.system)+'\n'
    return output

  def setupHelp(self, help):
    import nargs
    config.package.CMakePackage.setupHelp(self, help)
    help.addArgument('KOKKOS', '-download-kokkos-cxx-std-threads=<bool>',  nargs.ArgBool(None, False, 'Build kokkos for C++ threads'))
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.externalpackagesdir = framework.require('PETSc.options.externalpackagesdir',self)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.setCompilers    = framework.require('config.setCompilers', self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.flibs           = framework.require('config.packages.flibs',self)
    self.cxxlibs         = framework.require('config.packages.cxxlibs',self)
    self.mathlib         = framework.require('config.packages.mathlib',self)
    self.deps            = [self.blasLapack,self.flibs,self.cxxlibs,self.mathlib]
    self.openmp          = framework.require('config.packages.OpenMP',self)
    self.cuda            = framework.require('config.packages.CUDA',self)
    self.hip             = framework.require('config.packages.HIP',self)
    self.sycl            = framework.require('config.packages.SYCL',self)
    self.hwloc           = framework.require('config.packages.hwloc',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.odeps           = [self.mpi,self.openmp,self.hwloc,self.cuda,self.hip]
    return

  def versionToStandardForm(self,ver):
    '''Converts from kokkos 30101 notation to standard notation 3.1.01'''
    return ".".join(map(str,[int(ver)//10000, int(ver)//100%100, int(ver)%100]))

  # duplicate from Trilinos.py
  def toString(self,string):
    string    = self.libraries.toString(string)
    if self.requiresrpath: return string
    newstring = ''
    for i in string.split(' '):
      if i.find('-rpath') == -1:
        newstring = newstring+' '+i
    return newstring.strip()

  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DUSE_XSDK_DEFAULTS=YES')
    # Whether code deprecated in major release 4 is available
    args.append('-DKokkos_ENABLE_DEPRECATED_CODE_4=OFF')

    # always use C/C++'s alignment (i.e., sizeof(RealType)) for complex,
    # instead of Kokkos's default "alignas(2 * sizeof(RealType))"
    args.append('-DKokkos_ENABLE_COMPLEX_ALIGN=OFF')

    if not self.compilerFlags.debugging:
      args.append('-DXSDK_ENABLE_DEBUG=NO')

    if self.mpi.found and not self.mpi.usingMPIUni:
      args.append('-DKokkos_ENABLE_MPI=ON')

    if self.hwloc.found:
      args.append('-DKokkos_ENABLE_HWLOC=ON')
      args.append('-DKokkos_HWLOC_DIR='+self.hwloc.directory)

    self.system = ['Serial']
    args.append('-DKokkos_ENABLE_SERIAL=ON')
    if self.argDB['download-kokkos-cxx-std-threads']:
      args.append('-DKokkos_ENABLE_THREADS=ON')
      self.system.append('C++ Threads')
    if self.openmp.found:
      args.append('-DKokkos_ENABLE_OPENMP=ON')
      self.system.append('OpenMP')

    lang = 'cxx'
    deviceArchName = ''
    if self.cuda.found:
      # lang is used below to get nvcc C++ dialect. In a case with nvhpc-21.7 and "nvcc -ccbin nvc++ -std=c++17",
      # nvcc complains the host compiler does not support c++17, even though it does. So we have to respect
      # what nvcc thinks, instead of taking the c++ dialect directly from the host compiler.
      lang = 'cuda'
      args.append('-DKokkos_ENABLE_CUDA=ON')
      args.append('-DKokkos_ENABLE_CUDA_LAMBDA:BOOL=ON')
      # Use of cudaMallocAsync() is turned off by default since Kokkos-4.5.0, see https://github.com/kokkos/kokkos/pull/7353,
      # since it interferes with the CUDA aware MPI. We also turn it off for older versions.
      args.append('-DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC:BOOL=OFF')
      self.system.append('CUDA')
      self.pushLanguage('CUDA')
      petscNvcc = self.getCompiler()
      cudaFlags = self.updatePackageCUDAFlags(self.getCompilerFlags())
      self.popLanguage()
      args = self.rmArgsStartsWith(args, '-DCMAKE_CUDA_COMPILER')
      args = self.rmArgsStartsWith(args, '-DCMAKE_CUDA_FLAGS')
      args = self.rmArgsStartsWith(args, '-DCMAKE_C_COMPILER')
      args = self.rmArgsStartsWith(args, '-DCMAKE_C_FLAGS')
      args = self.rmArgsStartsWith(args, '-DCMAKE_CXX_COMPILER')
      if self.cuda.cudaclang:
        args = self.rmArgsStartsWith(args, '-DCMAKE_CXX_FLAGS')
        args.append('-DCMAKE_CXX_COMPILER="'+petscNvcc+'"')
        args.append('-DCMAKE_CXX_FLAGS="'+cudaFlags.replace('-x cuda', '')+'"')
      else:
        args.append('-DKOKKOS_CUDA_OPTIONS="'+cudaFlags.replace(' ', ';')+'"')
        args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_COMPILER=')
        # Kokkos passes the C++ compiler flags to nvcc which barfs with
        # nvcc fatal   : 'g': expected a number
        args = [a.replace('-Og', '-O1') if a.startswith('-DCMAKE_CXX_FLAG') else a for a in args]
        args.append('-DCMAKE_CXX_COMPILER='+self.getCompiler('Cxx')) # use the host CXX compiler, let Kokkos handle the nvcc_wrapper business
        #  Kokkos nvcc_wrapper REQUIRES nvcc be visible in the PATH!
        path = os.getenv('PATH')
        nvccpath = os.path.dirname(petscNvcc)
        if nvccpath:
          # Put nvccpath in the beginning of PATH, as there might be other nvcc in PATH and we got this one from --with-cuda-dir.
          # Kokkos provides Kokkos_CUDA_DIR and CUDA_ROOT. But they do not actually work (as of Jan. 2022) in the aforementioned
          # case, since these cmake options are not passed correctly to nvcc_wrapper.
          os.environ['PATH'] = nvccpath+':'+path
      if hasattr(self.cuda,'cudaArch'):
        # See https://developer.nvidia.com/cuda-gpus and https://en.wikipedia.org/wiki/CUDA#GPUs_supported.
        # But Kokkos only supports some of them, see https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#nvidia-gpus,
        nameToGens = {'KEPLER':   ['30', '32', '35', '37'],
                      'MAXWELL':  ['50', '52', '53'],
                      'PASCAL':   ['60', '61'],
                      'VOLTA':    ['70', '72'],
                      'TURING':   ['75'],
                      'AMPERE':   ['80', '86'],
                      'ADA':      ['89'],
                      'HOPPER':   ['90'],
                      'BLACKWELL':['100', '120']}
        gen = self.cuda.cudaArchSingle() # cudaArchSingle() returns a number, such as '75' or '120'
        foundName = None
        for name, gens in nameToGens.items():
          if gen in gens:
            foundName = name
            break
        if foundName:
          # Kokkos uses names like VOLTA70, AMPERE86
          deviceArchName = foundName + self.cuda.cudaArchSingle()
        else:
          raise RuntimeError('Could not find a Kokkos arch name for CUDA gen number '+ self.cuda.cudaArchSingle())
      else:
        raise RuntimeError('You must set --with-cuda-arch=60, 70, 75, 80 etc.')
    elif self.hip.found:
      lang = 'hip'
      self.system.append('HIP')
      args.append('-DKokkos_ENABLE_HIP=ON')
      with self.Language('HIP'):
        petscHipc = self.getCompiler()
        hipFlags = self.updatePackageCxxFlags(self.getCompilerFlags())
        # kokkos uses clang and offload flag
        hipFlags = ' '.join([i for i in hipFlags.split() if '--amdgpu-target' not in i])
      args.append('-DKOKKOS_HIP_OPTIONS="'+hipFlags.replace(' ',';')+'"')
      self.getExecutable(petscHipc,getFullPath=1,resultName='systemHipc')
      if not hasattr(self,'systemHipc'):
        raise RuntimeError('HIP error: could not find path of hipcc')
      args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_COMPILER=')
      args.append('-DCMAKE_CXX_COMPILER='+self.systemHipc)
      args = self.rmArgsStartsWith(args, '-DCMAKE_CXX_FLAGS')
      args.append('-DCMAKE_CXX_FLAGS="' + hipFlags + '"')
      # See https://kokkos.org/kokkos-core-wiki/keywords.html#amd-gpus, AMD_GFX is preferred over VEGA
      deviceArchName = 'AMD_' + self.hip.hipArch.upper()
      if self.hip.unifiedMemory: deviceArchName += '_APU'
      args.append('-DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=OFF')
    elif self.sycl.found:
      lang = 'sycl'
      self.system.append('SYCL')
      args.append('-DKokkos_ENABLE_SYCL=ON')
      with self.Language('SYCL'):
        petscSyclc = self.getCompiler()
      self.getExecutable(petscSyclc,getFullPath=1,resultName='systemSyclc')
      if not hasattr(self,'systemSyclc'):
        raise RuntimeError('SYCL error: could not find path of the sycl compiler')
      args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_COMPILER=')
      args.append('-DCMAKE_CXX_COMPILER='+self.systemSyclc)
      args.append('-DCMAKE_CXX_EXTENSIONS=OFF')
      args.append('-DKokkos_ENABLE_DEPRECATED_CODE_3=OFF')
      if hasattr(self.sycl,'syclArch') and self.sycl.syclArch != 'x86_64':
        deviceArchName = 'INTEL_' + self.sycl.syclArch.upper()  # Ex. map xehp to INTEL_XEHP

    if deviceArchName: args.append('-DKokkos_ARCH_'+deviceArchName+'=ON')

    langdialect = getattr(self.setCompilers,lang+'dialect',None)
    if langdialect:
      # langdialect is only set as an attribute if the user specifically chose a dialect
      # (see config/setCompilers.py::checkCxxDialect())
      args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_STANDARD=')
      args.append('-DCMAKE_CXX_STANDARD='+langdialect[-2:]) # e.g., extract 14 from C++14
    return args

  def configureLibrary(self):
    import os
    if self.cuda.found:
      self.buildLanguages = ['CUDA']
      oldFlags = self.setCompilers.CUDAPPFLAGS
      if self.cuda.cudaclang:
        self.addMakeMacro('KOKKOS_USE_CUDACLANG_COMPILER',1) # use the clang compiler to compile PETSc Kokkos code
      else:
        self.addMakeMacro('KOKKOS_USE_CUDA_COMPILER',1) # use the CUDA compiler to compile PETSc Kokkos code
        self.setCompilers.CUDAPPFLAGS += " -ccbin " + self.getCompiler('Cxx')
    elif self.hip.found:
      self.buildLanguages= ['HIP']
      self.addMakeMacro('KOKKOS_USE_HIP_COMPILER',1)  # use the HIP compiler to compile PETSc Kokkos code
    elif self.sycl.found:
      self.buildLanguages= ['SYCL']
      self.setCompilers.SYCLPPFLAGS        += " -Wno-deprecated-declarations "
      self.setCompilers.SYCLFLAGS          += ' -fno-sycl-id-queries-fit-in-int -fsycl-unnamed-lambda '
      self.addMakeMacro('KOKKOS_USE_SYCL_COMPILER',1)
    else:
      self.addDefine('HAVE_KOKKOS_WITHOUT_GPU', 1) # Kokkos is used without GPUs (i.e., host only)

    config.package.CMakePackage.configureLibrary(self)

    if self.cuda.found:
      self.addMakeMacro('KOKKOS_BIN',os.path.join(self.directory,'bin'))
      self.logWrite('Checking if Kokkos is configured with CUDA lambda\n')
      self.pushLanguage('CUDA')
      cuda_lambda_test = '''
         #include <Kokkos_Macros.hpp>
         #if !defined(KOKKOS_ENABLE_CUDA_LAMBDA)
         #error "Kokkos is not configured with CUDA lambda"
         #endif
      '''

      self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.include)
      if self.checkPreprocess(cuda_lambda_test):
        self.logPrint('Kokkos is configured with CUDA lambda\n')
      else:
        raise RuntimeError('Kokkos is not configured with -DKokkos_ENABLE_CUDA_LAMBDA. PETSc usage requires Kokkos to be configured with that')
      self.setCompilers.CUDAPPFLAGS = oldFlags
      self.popLanguage()
