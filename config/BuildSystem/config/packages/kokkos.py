import config.package
import os

class Configure(config.package.CMakePackage):
  def __init__(self, framework):
    config.package.CMakePackage.__init__(self, framework)
    self.gitcommit        = 'd4ed86ffb1b156faaa0f936ce839cfd6d3282478' # origin/develop of 2021-04-28
    self.versionname      = 'KOKKOS_VERSION'
    self.download         = ['git://https://github.com/kokkos/kokkos.git']
    self.downloaddirnames = ['kokkos']
    self.excludedDirs     = ['kokkos-kernels'] # Do not wrongly think kokkos-kernels as kokkos-vernum
    #TODO: We should use CUDAC (instead of CXX) to validate the headers. Using CXX does not work with newer Kokkos.
    #self.includes         = ['Kokkos_Macros.hpp']
    self.liblist          = [['libkokkoscontainers.a','libkokkoscore.a']]
    self.functions        = ['']
    self.functionsCxx     = [1,'namespace Kokkos {void initialize(int&,char*[]);}','int one = 1;char* args[1];Kokkos::initialize(one,args);']
    self.cxx              = 1
    self.downloadonWindows= 0
    self.hastests         = 1
    self.requiresrpath    = 1
    self.precisions       = ['single','double']
    self.kokkos_cxxdialect = 'C++14' # requirement for which compiler is used to compile Kokkos
    return

  def __str__(self):
    output  = config.package.CMakePackage.__str__(self)
    if hasattr(self,'system'): output += '  Backend: '+self.system+'\n'
    return output

  def setupHelp(self, help):
    import nargs
    config.package.CMakePackage.setupHelp(self, help)
    help.addArgument('KOKKOS', '-with-kokkos-cuda-arch', nargs.ArgString(None, 0, 'One of KEPLER30, KEPLER32, KEPLER35, KEPLER37, MAXWELL50, MAXWELL52, MAXWELL53, PASCAL60, PASCAL61, VOLTA70, VOLTA72, TURING75, AMPERE80, use nvidia-smi'))
    help.addArgument('KOKKOS', '-with-kokkos-hip-arch',  nargs.ArgString(None, 0, 'One of VEGA900, VEGA906, VEGA908'))
    return

  def setupDependencies(self, framework):
    config.package.CMakePackage.setupDependencies(self, framework)
    self.externalpackagesdir = framework.require('PETSc.options.externalpackagesdir',self)
    self.compilerFlags   = framework.require('config.compilerFlags', self)
    self.blasLapack      = framework.require('config.packages.BlasLapack',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.flibs           = framework.require('config.packages.flibs',self)
    self.cxxlibs         = framework.require('config.packages.cxxlibs',self)
    self.mathlib         = framework.require('config.packages.mathlib',self)
    self.deps            = [self.mpi,self.blasLapack,self.flibs,self.cxxlibs,self.mathlib]
    self.openmp          = framework.require('config.packages.openmp',self)
    self.pthread         = framework.require('config.packages.pthread',self)
    self.cuda            = framework.require('config.packages.cuda',self)
    self.hip             = framework.require('config.packages.hip',self)
    self.hwloc           = framework.require('config.packages.hwloc',self)
    self.mpi             = framework.require('config.packages.MPI',self)
    self.odeps           = [self.openmp,self.hwloc,self.cuda,self.hip,self.pthread]
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
    if not self.compilerFlags.debugging:
      args.append('-DXSDK_ENABLE_DEBUG=NO')

    if self.checkSharedLibrariesEnabled():
      args.append('-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON')
      args.append('-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=ON')
    args.append('-DCMAKE_CXX_STANDARD=' + self.compilers.cxxdialect[-2:])

    if self.mpi.found:
      args.append('-DKokkos_ENABLE_MPI=ON')

    if self.hwloc.found:
      args.append('-DKokkos_ENABLE_HWLOC=ON')
      args.append('-DKokkos_HWLOC_DIR='+self.hwloc.directory)

    # looks for pthread by default so need to turn it off unless specifically requested
    pthreadfound = self.pthread.found
    if not 'with-pthread' in self.framework.clArgDB:
      pthreadfound = 0

    if self.openmp.found + pthreadfound + self.cuda.found > 1:
      raise RuntimeError("Kokkos only supports a single parallel system during its configuration")

    args.append('-DKokkos_ENABLE_SERIAL=ON')
    if self.openmp.found:
      args.append('-DKokkos_ENABLE_OPENMP=ON')
      self.system = 'OpenMP'
    if pthreadfound:
      args.append('-DKokkos_ENABLE_PTHREAD=ON')
      self.system = 'PThread'

    if self.cuda.found:
      lang = 'cuda'
      args.append('-DKokkos_ENABLE_CUDA=ON')
      self.system = 'CUDA'
      self.pushLanguage('CUDA')
      petscNvcc = self.getCompiler()
      cudaFlags = self.getCompilerFlags()
      self.popLanguage()
      args.append('-DKOKKOS_CUDA_OPTIONS="'+cudaFlags.replace(' ',';')+'"')
      # Kokkos must be compiled with its horrible nvcc_wrapper script when using nvcc
      # cannot find way to set nvcc exectuable
      # NVCC_WRAPPER_DEFAULT_COMPILER
      args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_COMPILER=')
      dir = self.externalpackagesdir.dir
      args.append('-DCMAKE_CXX_COMPILER='+os.path.join(dir,'git.kokkos','bin','nvcc_wrapper'))
      if 'with-kokkos-cuda-arch' in self.framework.clArgDB:
        gencodearch = self.argDB['with-kokkos-cuda-arch']
      else:
        genToName = {'30' : 'KEPLER30','32' : 'KEPLER32', '35' : 'KEPLER35', '37' : 'KEPLER37', '50': 'MAXWELL50', '52': 'MAXWELL52', '53' : 'MAXWELL53', '60' : 'PASCAL60', '61' : 'PASCAL61', '70' : 'VOLTA70', '72': 'VOLTA72', '75' : 'TURING75', '80' : 'AMPERE80'}
        if hasattr(self.cuda,'gencodearch'):
          gencodearch = genToName[self.cuda.gencodearch]
        else:
          raise RuntimeError('You must set -with-kokkos-cuda-arch=PASCAL61, VOLTA70, VOLTA72, TURING75 etc.')
      args.append('-DKokkos_ARCH_'+gencodearch+'=ON')
      args.append('-DKokkos_ENABLE_CUDA_LAMBDA:BOOL=ON')
      #  Kokkos nvcc_wrapper REQUIRES nvcc be visible in the PATH!
      path = os.getenv('PATH')
      nvccpath = os.path.dirname(petscNvcc)
      if nvccpath:
         os.environ['PATH'] = path+':'+nvccpath
    elif self.hip.found:
      lang = 'hip'
      self.system = 'HIP'
      args.append('-DKokkos_ENABLE_HIP=ON')
      with self.Language('HIP'):
        petscHipc = self.getCompiler()
        hipFlags = self.updatePackageCFlags(self.getCompilerFlags())
      args.append('-DKOKKOS_HIP_OPTIONS="'+hipFlags.replace(' ',';')+'"')
      self.getExecutable(petscHipc,getFullPath=1,resultName='systemHipc')
      if not hasattr(self,'systemHipc'):
        raise RuntimeError('HIP error: could not find path of hipcc')
      args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_COMPILER=')
      args.append('-DCMAKE_CXX_COMPILER='+self.systemHipc)
      args = self.rmArgsStartsWith(args, '-DCMAKE_CXX_FLAGS')
      args.append('-DCMAKE_CXX_FLAGS="' + hipFlags + '"')
      if not 'with-kokkos-hip-arch' in self.framework.clArgDB:
        raise RuntimeError('You must set -with-kokkos-hip-arch=VEGA900, VEGA906, VEGA908 etc.')
      args.append('-DKokkos_ARCH_'+self.argDB['with-kokkos-hip-arch']+'=ON')
      args.append('-DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=OFF')
    else:
      lang = 'cxx'

    # set -DCMAKE_CXX_STANDARD=
    if not hasattr(self.compilers,lang+'dialect'):
      raise RuntimeError('Could not determine C++ dialect for the '+lang.upper()+' Compiler')
    else:
      langdialect = getattr(self.compilers,lang+'dialect')
      if langdialect < self.kokkos_cxxdialect:
        raise RuntimeError('Kokkos requires '+self.kokkos_cxxdialect+' but the '+lang.upper()+ 'compiler only supports '+langdialect)
      else:
        args = self.rmArgsStartsWith(args,'-DCMAKE_CXX_STANDARD=')
        args.append('-DCMAKE_CXX_STANDARD="' + langdialect.split("C++",1)[1] + '"') # e.g., extract 14 from C++14

    return args

  def configureLibrary(self):
    import os
    config.package.CMakePackage.configureLibrary(self)
    if self.cuda.found:
      self.addMakeMacro('KOKKOS_BIN',os.path.join(self.directory,'bin'))
      self.addMakeMacro('KOKKOS_USE_CUDA_COMPILER',1) # use the CUDA compiler to compile PETSc Kokkos code
    elif self.hip.found:
      self.addMakeMacro('KOKKOS_USE_HIP_COMPILER',1)  # use the HIP compiler to compile PETSc Kokkos code

