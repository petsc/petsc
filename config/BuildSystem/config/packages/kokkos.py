import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = '3.6.00'
    self.minversion       = '3.5.00'
    self.versionname      = 'KOKKOS_VERSION'
    self.download         = ['git://https://github.com/kokkos/kokkos.git']
    self.downloaddirnames = ['kokkos']
    self.excludedDirs     = ['kokkos-kernels'] # Do not wrongly think kokkos-kernels as kokkos-vernum
    self.includes         = ['Kokkos_Macros.hpp']
    self.liblist          = [['libkokkoscontainers.a','libkokkoscore.a']]
    self.functions        = ['']
    self.functionsCxx     = [1,'namespace Kokkos {void initialize(int&,char*[]);}','int one = 1;char* args[1];Kokkos::initialize(one,args);']
    self.minCxxVersion    = 'c++14'
    self.buildLanguages   = ['Cxx'] # Depending on if cuda, hip or sycl is avaiable, it will be modified.
    self.hastests         = 1
    self.requiresrpath    = 1
    self.precisions       = ['single','double']
    self.devicePackage    = 1
    self.minCmakeVersion  = (3,16,0)
    return

  def __str__(self):
    output  = config.package.CMakePackage.__str__(self)
    if hasattr(self,'system'): output += '  Backend: '+self.system+'\n'
    return output

  def setupHelp(self, help):
    import nargs
    config.package.CMakePackage.setupHelp(self, help)
    help.addArgument('KOKKOS', '-with-kokkos-init-warnings=<bool>',  nargs.ArgBool(None, True, 'Enable/disable warnings in Kokkos initialization'))
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
    self.openmp          = framework.require('config.packages.openmp',self)
    self.pthread         = framework.require('config.packages.pthread',self)
    self.cuda            = framework.require('config.packages.cuda',self)
    self.hip             = framework.require('config.packages.hip',self)
    self.sycl            = framework.require('config.packages.sycl',self)
    self.hwloc           = framework.require('config.packages.hwloc',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.odeps           = [self.mpi,self.openmp,self.hwloc,self.cuda,self.hip,self.pthread]
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

    # always use C/C++'s alignment (i.e., sizeof(RealType)) for complex,
    # instead of Kokkos's default "alignas(2 * sizeof(RealType))"
    args.append('-DKokkos_ENABLE_COMPLEX_ALIGN=OFF')

    if not self.compilerFlags.debugging:
      args.append('-DXSDK_ENABLE_DEBUG=NO')

    if self.checkSharedLibrariesEnabled():
      args.append('-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON')
      args.append('-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=ON')

    if self.mpi.found:
      args.append('-DKokkos_ENABLE_MPI=ON')

    if self.hwloc.found:
      args.append('-DKokkos_ENABLE_HWLOC=ON')
      args.append('-DKokkos_HWLOC_DIR='+self.hwloc.directory)

    # looks for pthread by default so need to turn it off unless specifically requested
    pthreadfound = self.pthread.found
    if not 'with-pthread' in self.framework.clArgDB:
      pthreadfound = 0

    args.append('-DKokkos_ENABLE_SERIAL=ON')
    args.append('-DKokkos_ENABLE_LIBDL=OFF')
    if self.openmp.found:
      args.append('-DKokkos_ENABLE_OPENMP=ON')
      self.system = 'OpenMP'
    if pthreadfound:
      args.append('-DKokkos_ENABLE_PTHREAD=ON')
      self.system = 'PThread'

    lang = 'cxx'
    deviceArchName = ''
    if self.cuda.found:
      # lang is used below to get nvcc C++ dialect. In a case with nvhpc-21.7 and "nvcc -ccbin nvc++ -std=c++17",
      # nvcc complains the host compiler does not support c++17, even though it does. So we have to respect
      # what nvcc thinks, instead of taking the c++ dialect directly from the host compiler.
      lang = 'cuda'
      args.append('-DKokkos_ENABLE_CUDA=ON')
      self.system = 'CUDA'
      self.pushLanguage('CUDA')
      petscNvcc = self.getCompiler()
      cudaFlags = self.getCompilerFlags()
      self.popLanguage()
      args.append('-DKOKKOS_CUDA_OPTIONS="'+cudaFlags.replace(' ',';')+'"')
      args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_COMPILER=')
      args.append('-DCMAKE_CXX_COMPILER='+self.getCompiler('Cxx')) # use the host CXX compiler, let Kokkos handle the nvcc_wrapper business
      genToName = {'3': 'KEPLER','5': 'MAXWELL', '6': 'PASCAL', '7': 'VOLTA', '8': 'AMPERE', '9': 'LOVELACE', '10': 'HOPPER'}
      if hasattr(self.cuda,'cudaArch'):
        generation = self.cuda.cudaArch[:-1] # cudaArch is a number 'nn', such as '75'
        try:
          # Kokkos uses names like VOLTA75, AMPERE86
          deviceArchName = genToName[generation] + self.cuda.cudaArch
        except KeyError:
          raise RuntimeError('Could not find an arch name for CUDA gen number '+ self.cuda.cudaArch)
      else:
        raise RuntimeError('You must set --with-cuda-arch=60, 70, 75, 80 etc.')
      args.append('-DKokkos_ENABLE_CUDA_LAMBDA:BOOL=ON')
      #  Kokkos nvcc_wrapper REQUIRES nvcc be visible in the PATH!
      path = os.getenv('PATH')
      nvccpath = os.path.dirname(petscNvcc)
      if nvccpath:
        # Put nvccpath in the beginning of PATH, as there might be other nvcc in PATH and we got this one from --with-cuda-dir.
        # Kokkos provids Kokkos_CUDA_DIR and CUDA_ROOT. But they do not actually work (as of Jan. 2022) in the aforementioned
        # case, since these cmake options are not passed correctly to nvcc_wrapper.
        os.environ['PATH'] = nvccpath+':'+path
    elif self.hip.found:
      lang = 'hip'
      self.system = 'HIP'
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
      deviceArchName = self.hip.hipArch.upper().replace('GFX','VEGA',1) # ex. map gfx90a to VEGA90A
      args.append('-DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=OFF')
    elif self.sycl.found:
      lang = 'sycl'
      self.system = 'SYCL'
      args.append('-DKokkos_ENABLE_SYCL=ON')
      with self.Language('SYCL'):
        petscSyclc = self.getCompiler()
        syclFlags = self.updatePackageCxxFlags(self.getCompilerFlags())
      self.getExecutable(petscSyclc,getFullPath=1,resultName='systemSyclc')
      if not hasattr(self,'systemSyclc'):
        raise RuntimeError('SYCL error: could not find path of the sycl compiler')
      args = self.rmArgsStartsWith(args, '-DCMAKE_CXX_FLAGS')
      args.append('-DCMAKE_CXX_FLAGS="' + syclFlags.replace('"','\\"') + '"')
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
      self.addMakeMacro('KOKKOS_USE_CUDA_COMPILER',1) # use the CUDA compiler to compile PETSc Kokkos code
    elif self.hip.found:
      self.buildLanguages= ['HIP']
      self.addMakeMacro('KOKKOS_USE_HIP_COMPILER',1)  # use the HIP compiler to compile PETSc Kokkos code
    elif self.sycl.found:
      self.buildLanguages= ['SYCL']
      self.setCompilers.SYCLPPFLAGS        += " -Wno-deprecated-declarations "
      self.setCompilers.SYCLFLAGS          += ' -fno-sycl-id-queries-fit-in-int -fsycl-unnamed-lambda '
      self.addMakeMacro('KOKKOS_USE_SYCL_COMPILER',1)

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
      oldFlags = self.compilers.CUDAPPFLAGS
      self.compilers.CUDAPPFLAGS += ' '+self.headers.toString(self.include)
      if self.checkPreprocess(cuda_lambda_test):
        self.logPrint('Kokkos is configured with CUDA lambda\n')
      else:
        raise RuntimeError('Kokkos is not configured with -DKokkos_ENABLE_CUDA_LAMBDA. PETSc usage requires Kokkos to be configured with that')
      self.compilers.CUDAPPFLAGS = oldFlags
      self.popLanguage()

    if self.argDB['with-kokkos-init-warnings']: # usually one wants to enable warnings
      self.addDefine('HAVE_KOKKOS_INIT_WARNINGS', 1)
