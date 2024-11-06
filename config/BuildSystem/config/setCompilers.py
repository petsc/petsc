from __future__ import generators
import config.base
import config
import os
import contextlib
from functools import reduce
from collections import namedtuple
from collections import defaultdict

# not sure how to handle this with 'self' so it's outside the class
def noCheck(command, status, output, error):
  return

try:
  any
except NameError:
  def any(lst):
    return reduce(lambda x,y:x or y,lst,False)

def _picTestIncludes(export=''):
  return '\n'.join(['#include <stdio.h>',
                    'int (*fprintf_ptr)(FILE*,const char*,...) = fprintf;',
                    'int '+export+' foo(void){',
                    '  fprintf_ptr(stdout,"hello");',
                    '  return 0;',
                    '}',
                    'void bar(void){foo();}\n'])


isUname_value          = False
isLinux_value          = False
isCygwin_value         = False
isSolaris_value        = False
isDarwin_value         = False
isDarwinCatalina_value = False
isFreeBSD_value        = False
isARM_value            = -1

class CaseInsensitiveDefaultDict(defaultdict):
  __slots__ = ()

  def update(self,*args):
    for x in args:
      for key,val in x.items():
        self[key] = val

  def __setitem__(self,key,val):
    if not isinstance(key,str):
      raise RuntimeError('must use strings as keys for {cls}'.format(cls=self.__class__))
    # super() without args is python3 only
    super(defaultdict,self).__setitem__(key.lower(),val)

  def __missing__(self,key):
    if not isinstance(key,str):
      raise RuntimeError('must use strings as keys for {cls}'.format(cls=self.__class__))
    key = key.lower()
    if key not in self.keys():
      self[key] = self.default_factory()
    return self[key]

def default_cxx_dialect_ranges():
  return ('c++11','c++20')

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix            = ''
    self.substPrefix             = ''
    self.usedMPICompilers        = 0
    self.mainLanguage            = 'C'
    self.cxxDialectRange         = CaseInsensitiveDefaultDict(default_cxx_dialect_ranges)
    self.cxxDialectPackageRanges = ({},{})
    return

  def __str__(self):
    self.compilerflags = self.framework.getChild('config.compilerFlags')
    desc = ['Compilers:']
    if hasattr(self, 'CC'):
      self._setupCompiler('C',desc)
    if hasattr(self, 'CUDAC'):
      self._setupCompiler('CUDA',desc)
    if hasattr(self, 'HIPC'):
      self._setupCompiler('HIP',desc)
    if hasattr(self, 'SYCLC'):
      self._setupCompiler('SYCL',desc)
    if hasattr(self, 'CXX'):
      self._setupCompiler('Cxx',desc)
    if hasattr(self, 'FC'):
      self._setupCompiler('FC',desc)
    desc.append('Linkers:')
    if hasattr(self, 'staticLinker'):
      desc.append('  Static linker:   '+self.getSharedLinker()+' '+self.AR_FLAGS)
    elif hasattr(self, 'sharedLinker'):
      desc.append('  Shared linker:   '+self.getSharedLinker()+' '+self.getSharedLinkerFlags())
    if hasattr(self, 'dynamicLinker'):
      desc.append('  Dynamic linker:   '+self.getDynamicLinker()+' '+self.getDynamicLinkerFlags())
      desc.append('  Libraries linked against:   '+self.LIBS)
    return '\n'.join(desc)+'\n'

  def _setupCompiler(self,compiler,desc):
    """ Simple utility routine to minimize verbiage"""
    clabel='  '+compiler+' '
    if compiler == 'Cxx': clabel='  C++ '
    if compiler == 'FC':  clabel='  Fortran '
    self.pushLanguage(compiler)
    desc.append(clabel+'Compiler:         '
                +self.getCompiler()+' '+self.getCompilerFlags())
    if self.compilerflags.version[compiler]:
      desc.append('    Version: '+self.compilerflags.version[compiler])
    if not self.getLinker() == self.getCompiler():
      desc.append(clabel+'Linker:           '
                  +self.getLinker()+' '+self.getLinkerFlags())
    self.popLanguage()
    return

  def setupHelp(self, help):
    import nargs

    help.addArgument('Compilers', '-with-cpp=<prog>', nargs.Arg(None, None, 'Specify the C preprocessor'))
    help.addArgument('Compilers', '-CPP=<prog>',            nargs.Arg(None, None, 'Specify the C preprocessor'))
    help.addArgument('Compilers', '-CPPFLAGS=<string>',     nargs.Arg(None, None, 'Specify the C only (not used for C++ or FC) preprocessor options'))
    help.addArgument('Compilers', '-with-cc=<prog>',  nargs.Arg(None, None, 'Specify the C compiler'))
    help.addArgument('Compilers', '-CC=<prog>',             nargs.Arg(None, None, 'Specify the C compiler'))
    help.addArgument('Compilers', '-CFLAGS=<string>',       nargs.Arg(None, None, 'Overwrite the default PETSc C compiler flags\n\
       Use CFLAGS+= to add to (instead of replacing) the default flags'))
    help.addArgument('Compilers', '-CFLAGS+=<string>',      nargs.Arg(None, None, 'Add to the default PETSc C compiler flags'))
    help.addArgument('Compilers', '-CC_LINKER_FLAGS=<string>',  nargs.Arg(None, [], 'Specify the C linker flags'))

    help.addArgument('Compilers', '-CXXPP=<prog>',          nargs.Arg(None, None, 'Specify the C++ preprocessor'))
    help.addArgument('Compilers', '-CXXPPFLAGS=<string>',   nargs.Arg(None, None, 'Specify the C++ preprocessor options'))
    help.addArgument('Compilers', '-with-cxx=<prog>', nargs.Arg(None, None, 'Specify the C++ compiler'))
    help.addArgument('Compilers', '-CXX=<prog>',            nargs.Arg(None, None, 'Specify the C++ compiler'))
    help.addArgument('Compilers', '-CXXFLAGS=<string>',     nargs.Arg(None, None, 'Overwrite the default PETSc C++ compiler flags, also passed to linker\n\
       Use CXXFLAGS+ to add to (instead of replacing) the default flags'))
    help.addArgument('Compilers', '-CXXFLAGS+=<string>',    nargs.Arg(None, None, 'Add to the default PETSc C++ compiler flags, also passed to linker'))
    help.addArgument('Compilers', '-CXX_CXXFLAGS=<string>', nargs.Arg(None, '',   'Specify the C++ compiler-only options, not passed to linker'))
    help.addArgument('Compilers', '-CXX_LINKER_FLAGS=<string>',       nargs.Arg(None, [], 'Specify the C++ linker flags'))

    help.addArgument('Compilers', '-FPP=<prog>',            nargs.Arg(None, None, 'Specify the Fortran preprocessor'))
    help.addArgument('Compilers', '-FPPFLAGS=<string>',     nargs.Arg(None, None, 'Specify the Fortran preprocessor options'))
    help.addArgument('Compilers', '-with-fc=<prog>',  nargs.Arg(None, None, 'Specify the Fortran compiler'))
    help.addArgument('Compilers', '-FC=<prog>',             nargs.Arg(None, None, 'Specify the Fortran compiler'))
    help.addArgument('Compilers', '-FFLAGS=<string>',       nargs.Arg(None, None, 'Overwrite the default PETSc Fortran compiler flags\n\
       Use FFLAGS+= to add to (instead of replacing) the default flags'))
    help.addArgument('Compilers', '-FFLAGS+=<string>',      nargs.Arg(None, None, 'Add to the default PETSc Fortran compiler flags'))
    help.addArgument('Compilers', '-FC_LINKER_FLAGS=<string>',        nargs.Arg(None, [], 'Specify the FC linker flags'))

    help.addArgument('Compilers', '-with-large-file-io=<bool>', nargs.ArgBool(None, 0, 'Allow IO with files greater then 2 GB'))

    help.addArgument('Compilers', '-CUDAPP=<prog>',        nargs.Arg(None, None, 'Specify the CUDA preprocessor'))
    help.addArgument('Compilers', '-CUDAPPFLAGS=<string>', nargs.Arg(None, None, 'Specify the CUDA preprocessor options'))
    help.addArgument('Compilers', '-with-cudac=<prog>',    nargs.Arg(None, None, 'Specify the CUDA compiler'))
    help.addArgument('Compilers', '-CUDAC=<prog>',         nargs.Arg(None, None, 'Specify the CUDA compiler'))
    help.addArgument('Compilers', '-CUDAFLAGS=<string>',   nargs.Arg(None, None, 'Overwrite the PETSc default CUDA compiler flags\n\
       Use CUDAFLAGS+= to add to (instead of replacing) the default flags'))
    help.addArgument('Compilers', '-CUDAC_LINKER_FLAGS=<string>',        nargs.Arg(None, [], 'Specify the CUDA linker flags'))

    help.addArgument('Compilers', '-HIPPP=<prog>',        nargs.Arg(None, None, 'Specify the HIP preprocessor'))
    help.addArgument('Compilers', '-HIPPPFLAGS=<string>', nargs.Arg(None, None, 'Specify the HIP preprocessor options'))
    help.addArgument('Compilers', '-with-hipc=<prog>',    nargs.Arg(None, None, 'Specify the HIP compiler'))
    help.addArgument('Compilers', '-HIPC=<prog>',         nargs.Arg(None, None, 'Specify the HIP compiler'))
    help.addArgument('Compilers', '-HIPFLAGS=<string>',   nargs.Arg(None, None, 'Overwrite the PETSc default HIP compiler flags\n\
       Use HIPFLAGS+= to add to (instead of replacing) the default flags'))
    help.addArgument('Compilers', '-HIPC_LINKER_FLAGS=<string>',        nargs.Arg(None, [], 'Specify the HIP linker flags'))

    help.addArgument('Compilers', '-SYCLPP=<prog>',        nargs.Arg(None, None, 'Specify the SYCL preprocessor'))
    help.addArgument('Compilers', '-SYCLPPFLAGS=<string>', nargs.Arg(None, None, 'Specify the SYCL preprocessor options'))
    help.addArgument('Compilers', '-with-syclc=<prog>',    nargs.Arg(None, None, 'Specify the SYCL compiler'))
    help.addArgument('Compilers', '-SYCLC=<prog>',         nargs.Arg(None, None, 'Specify the SYCL compiler'))
    help.addArgument('Compilers', '-SYCLFLAGS=<string>',   nargs.Arg(None, None, 'Overwrite the PETSc default SYCL compiler flags\n\
       Use SYCLLAGS+= to add to (instead of replacing) the default flags'))
    help.addArgument('Compilers', '-SYCLC_LINKER_FLAGS=<string>',        nargs.Arg(None, '', 'Specify the SYCL linker flags'))

##    help.addArgument('Compilers', '-LD=<prog>',              nargs.Arg(None, None, 'Specify the executable linker'))
##    help.addArgument('Compilers', '-CC_LD=<prog>',           nargs.Arg(None, None, 'Specify the linker for C only'))
##    help.addArgument('Compilers', '-CXX_LD=<prog>',          nargs.Arg(None, None, 'Specify the linker for C++ only'))
##    help.addArgument('Compilers', '-FC_LD=<prog>',           nargs.Arg(None, None, 'Specify the linker for Fortran only'))
    help.addArgument('Compilers', '-with-shared-ld=<prog>',  nargs.Arg(None, None, 'Specify the shared linker'))
    help.addArgument('Compilers', '-LD_SHARED=<prog>',       nargs.Arg(None, None, 'Specify the shared linker'))
    help.addArgument('Compilers', '-LDFLAGS=<string>',       nargs.Arg(None, '',   'Specify the linker options'))
    help.addArgument('Compilers', '-with-ar=<prog>',                nargs.Arg(None, None,   'Specify the archiver'))
    help.addArgument('Compilers', '-AR=<prog>',                     nargs.Arg(None, None,   'Specify the archiver flags'))
    help.addArgument('Compilers', '-AR_FLAGS=<string>',               nargs.Arg(None, None,   'Specify the archiver flags'))
    help.addArgument('Compilers', '-with-ranlib=<prog>',            nargs.Arg(None, None,   'Specify ranlib'))
    help.addArgument('Compilers', '-with-pic=<bool>',               nargs.ArgBool(None, 0, 'Compile with -fPIC or equivalent flag if possible'))
    help.addArgument('Compilers', '-sharedLibraryFlags=<string>',     nargs.Arg(None, [], 'Specify the shared library flags'))
    help.addArgument('Compilers', '-dynamicLibraryFlags=<string>',    nargs.Arg(None, [], 'Specify the dynamic library flags'))
    help.addArgument('Compilers', '-LIBS=<string>',          nargs.Arg(None, None, 'Specify extra libraries for all links'))
    help.addArgument('Compilers', '-with-environment-variables=<bool>',nargs.ArgBool(None, 0, 'Use compiler variables found in environment'))
    min_ver, max_ver   = default_cxx_dialect_ranges()
    min_ver            = int(min_ver[2:])
    max_ver            = int(max_ver[2:])
    available_dialects = []
    while min_ver <= max_ver:
      available_dialects.append(min_ver)
      min_ver += 3

    available_dialects = ', '.join(map(str, available_dialects))
    help.addArgument('Compilers', '-with-cxx-dialect=<dialect>',nargs.Arg(None, 'auto', 'Dialect under which to compile C++ sources. Pass "c++17" to use "-std=c++17", "gnu++17" to use "-std=gnu++17" or pass just the number (e.g. "17") to have PETSc auto-detect gnu extensions. Pass "auto" to let PETSc auto-detect everything or "0" to use the compiler"s default. Available: ({}, auto, 0)'.format(available_dialects)))
    help.addArgument('Compilers', '-with-hip-dialect=<dialect>',nargs.Arg(None, 'auto', 'Dialect under which to compile HIP sources. If set should probably be equivalent to c++ dialect (see --with-cxx-dialect)'))
    help.addArgument('Compilers', '-with-cuda-dialect=<dialect>',nargs.Arg(None, 'auto', 'Dialect under which to compile CUDA sources. If set should probably be equivalent to c++ dialect (see --with-cxx-dialect)'))
    help.addArgument('Compilers', '-with-sycl-dialect=<dialect>',nargs.Arg(None, 'auto', 'Dialect under which to compile SYCL sources. If set should probably be equivalent to c++ dialect (see --with-cxx-dialect)'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.languages = framework.require('PETSc.options.languages', self)
    self.libraries = self.framework.getChild('config.libraries')
    self.headers   = self.framework.getChild('config.headers')
    return

  @staticmethod
  def isNAG(compiler, log):
    '''Returns true if the compiler is a NAG F90 compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' -V',checkCommand = noCheck, log = log)
      output = output + error
      found = any([s in output for s in ['NAGWare Fortran','The Numerical Algorithms Group Ltd']])
      if found:
        if log: log.write('Detected NAG Fortran compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isMINGW(compiler, log):
    '''Returns true if the compiler is a MINGW compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' -v',checkCommand = noCheck, log = log)
      output = output + error
      if output.find('w64-mingw32') >= 0:
        if log: log.write('Detected MINGW compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isGNU(compiler, log):
    '''Returns true if the compiler is a GNU compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help | head -n 20 ', log = log)
      output = output + error
      found = (any([s in output for s in ['www.gnu.org',
                                         'bugzilla.redhat.com',
                                         'gcc.gnu.org',
                                         'gcc version',
                                         '-print-libgcc-file-name',
                                         'passed on to the various sub-processes invoked by gcc',
                                         'passed on to the various sub-processes invoked by cc',
                                         'passed on to the various sub-processes invoked by gfortran',
                                         'passed on to the various sub-processes invoked by g++',
                                         'passed on to the various sub-processes invoked by c++',
                                         ]])
              and not any([s in output for s in ['Intel(R)',
                                                 'Unrecognised option --help passed to ld', # NAG f95 compiler
                                                 'IBM XL', # XL compiler
                                                 ]]))
      if not found and Configure.isCrayPEWrapper(compiler,log):
        (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version', log = log)
        found = any([s in output for s in ['(GCC)','GNU Fortran','gcc-','g++-']])
      if found:
        if log: log.write('Detected GNU compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isClang(compiler, log):
    '''Returns true if the compiler is a Clang/LLVM compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help | head -n 500', log = log, logOutputflg = False)
      output = output + error
      found = (any([s in output for s in ['Emit Clang AST']])
               and not any([s in output for s in ['Win32 Development Tool Front End']]))
      if found:
        if log: log.write('Detected CLANG compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isHIP(compiler, log):
    '''Returns true if the compiler is a HIP compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version', log = log)
      output = output + error
      if 'HIP version:' in output:
        if log: log.write('Detected HIP compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isOneAPI(compiler, log):
    '''Returns true if the compiler is an Intel oneAPI compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version', log = log)
      output = output + error
      found = any([s in output for s in ['Intel(R) oneAPI']])
      if found:
        if log: log.write('Detected Intel oneAPI compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isSYCL(compiler, log):
    '''Returns true if the compiler is a SYCL compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version', log = log)
      output = output + error
      # Currently we only tested Intel oneAPI DPC++. Expand the list as more sycl compilers are available
      found = any([s in output for s in ['oneAPI DPC++']])
      if found:
        if log: log.write('Detected SYCL compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isNVCC(compiler, log):
    '''Returns true if the compiler is a NVCC compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version', log = log)
      output = output + error
      if 'Cuda compiler driver' in output:
        if log: log.write('Detected NVCC compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isNVC(compiler, log):
    '''Returns true if the compiler is an NVIDIA (former PGI) compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version', log = log)
      output = output + error
      if 'NVIDIA Compilers and Tools' in output:
        if log: log.write('Detected NVIDIA compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isGcc110plus(compiler, log):
    '''returns true if the compiler is gcc-11.0.x or later'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version', log = log)
      output = output + error
      import re
      strmatch = re.match(r'gcc[-0-9]*\s+\(.*\)\s+(\d+)\.(\d+)',output)
      if strmatch:
        VMAJOR,VMINOR = strmatch.groups()
        if (int(VMAJOR),int(VMINOR)) >= (11,0):
          if log: log.write('Detected Gcc110plus compiler\n')
          return 1
      if log: log.write('Did not detect Gcc110plus compiler\n')
    except RuntimeError:
      if log: log.write('Did not detect Gcc110plus compiler due to exception\n')
      pass

  @staticmethod
  def isGfortran45x(compiler, log):
    '''returns true if the compiler is gfortran-4.5.x'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version', log = log)
      output = output + error
      import re
      if re.match(r'GNU Fortran \(.*\) (4.5.\d+|4.6.0 20100703)', output):
        if log: log.write('Detected GFortran45x compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isGfortran46plus(compiler, log):
    '''returns true if the compiler is gfortran-4.6.x or later'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version', log = log)
      output = output + error
      import re
      strmatch = re.match(r'GNU Fortran\s+\(.*\)\s+(\d+)\.(\d+)',output)
      if strmatch:
        VMAJOR,VMINOR = strmatch.groups()
        if (int(VMAJOR),int(VMINOR)) >= (4,6):
          if log: log.write('Detected GFortran46plus compiler\n')
          return 1
    except RuntimeError:
      pass

  @staticmethod
  def isGfortran47plus(compiler, log):
    '''returns true if the compiler is gfortran-4.7.x or later'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version', log = log)
      output = output + error
      import re
      strmatch = re.match(r'GNU Fortran\s+\(.*\)\s+(\d+)\.(\d+)',output)
      if strmatch:
        VMAJOR,VMINOR = strmatch.groups()
        if (int(VMAJOR),int(VMINOR)) >= (4,7):
          if log: log.write('Detected GFortran47plus compiler\n')
          return 1
    except RuntimeError:
      pass

  @staticmethod
  def isGfortran100plus(compiler, log):
    '''returns true if the compiler is gfortran-10.0.x or later'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version', log = log)
      output = output + error
      import re
      strmatch = re.match(r'GNU Fortran\s+\(.*\)\s+(\d+)\.(\d+)',output)
      if strmatch:
        VMAJOR,VMINOR = strmatch.groups()
        if (int(VMAJOR),int(VMINOR)) >= (10,0):
          if log: log.write('Detected GFortran100plus compiler\n')
          return 1
    except RuntimeError:
      pass

  @staticmethod
  def isGfortran8plus(compiler, log):
    '''returns true if the compiler is gfortran-8 or later'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version', log = log)
      output = output + error
      import re
      strmatch = re.match(r'GNU Fortran\s+\(.*\)\s+(\d+)\.(\d+)',output)
      if strmatch:
        VMAJOR,VMINOR = strmatch.groups()
        if (int(VMAJOR),int(VMINOR)) >= (8,0):
          if log: log.write('Detected GFortran8plus compiler\n')
          return 1
    except RuntimeError:
      pass

  @staticmethod
  def isG95(compiler, log):
    '''Returns true if the compiler is g95'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help | head -n 20', log = log)
      output = output + error
      if 'Unrecognised option --help passed to ld' in output:    # NAG f95 compiler
        return 0
      if 'http://www.g95.org' in output:
        if log: log.write('Detected g95 compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isCompaqF90(compiler, log):
    '''Returns true if the compiler is Compaq f90'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help | head -n 20', log = log)
      output = output + error
      if 'Unrecognised option --help passed to ld' in output:    # NAG f95 compiler
        return 0
      found = any([s in output for s in ['Compaq Visual Fortran','Digital Visual Fortran']])
      if found:
        if log: log.write('Detected Compaq Visual Fortran compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isSun(compiler, log):
    '''Returns true if the compiler is a Sun/Oracle compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' -V',checkCommand = noCheck, log = log)
      output = output + error
      found = any([s in output for s in [' Sun C ',' Sun C++ ', ' Sun Fortran ']])
      if found:
        if log: log.write('Detected Sun/Oracle compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isIBM(compiler, log):
    '''Returns true if the compiler is a IBM compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' -qversion', log = log)
      output = output + error
      if 'IBM XL' in output:
        if log: log.write('Detected IBM compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isIntel(compiler, log):
    '''Returns true if the compiler is a Intel compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help | head -n 80', log = log)
      output = output + error
      if 'Intel(R)' in output:
        if log: log.write('Detected Intel compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isCrayKNL(compiler, log):
    '''Returns true if the compiler is a compiler for KNL running on a Cray'''
    x = os.getenv('PE_PRODUCT_LIST')
    if x and x.find('CRAYPE_MIC-KNL') > -1:
      if log: log.write('Detected Cray KNL compiler\n')
      return 1

  @staticmethod
  def isCray(compiler, log):
    '''Returns true if the compiler is a Cray compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' -V', log = log)
      output = output + error
      found = any([s in output for s in ['Cray C ','Cray Standard C','Cray C++ ','Cray Fortran ']])
      if found:
        if log: log.write('Detected Cray compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isCrayPEWrapper(compiler, log):
    '''Returns true if the compiler is a Cray Programming Environment (PE) compiler wrapper'''
    try:
      # Cray PE compiler wrappers (e.g., cc) when invoked with --version option will complain when CRAY_CPU_TARGET is set to erroneous value, but Cray raw compilers (e.g., craycc) won't. So use this behavior to differentiate cc from craycc.
      canary_value = '5dde31d2'
      (output, error, status) = config.base.Configure.executeShellCommand(
        'CRAY_CPU_TARGET="%s" %s --version' % (canary_value, compiler),
        checkCommand=config.base.Configure.passCheckCommand,
        log=log,
      )
      output = output + error
      if output.find(canary_value) >= 0:
        if log:
          log.write('Detected Cray PE compiler wrapper\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isCrayVector(compiler, log):
    '''Returns true if the compiler is a Cray compiler for a Cray Vector system'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' -VV', log = log)
      output = output + error
      if not status and output.find('x86') >= 0:
        return 0
      elif not status:
        if log: log.write('Detected Cray vector compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isPGI(compiler, log):
    '''Returns true if the compiler is a PGI compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' -V',checkCommand = noCheck, log = log)
      output = output + error
      found = any([s in output for s in ['The Portland Group','PGI Compilers and Tools']])
      if found:
        if log: log.write('Detected PGI compiler\n')
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isNEC(compiler, log):
    '''Returns true if the compiler is a NEC compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version',checkCommand = noCheck, log = log)
      output = output + error
      if output.find('NEC Corporation') >= 0:
        if log: log.write('Detected NEC compiler\n')
        return 1
    except RuntimeError:
      pass

  @classmethod
  def isWindows(cls, compiler, log):
    '''Returns true if the compiler is a Windows compiler'''
    if cls.isCygwin(log):
      compiler = os.path.basename(compiler)
      if compiler.startswith('win32fe'):
        if log: log.write('Detected Microsoft Windows native compiler\n')
        return 1
    if log: log.write('Detected Non-Microsoft Windows native compiler\n')
    return 0

  @classmethod
  def isMSVC(cls, compiler, log):
    """
    Returns true if the compiler is MSVC. Does not distinguish between raw MSVC and win32fe + MSVC
    """
    output, error, _ = cls.executeShellCommand(compiler + ' --version', checkCommand=noCheck, log=log)
    output           = '\n'.join((output, error)).casefold()
    found            = all(
      sub.casefold() in output for sub in ('microsoft', 'c/c++ optimizing compiler')
    )
    if log:
      log.write('Detected MSVC\n' if found else 'Did not detect MSVC\n')
    return int(found)

  @staticmethod
  def isSolarisAR(ar, log):
    '''Returns true AR is solaris'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(ar + ' -V',checkCommand = noCheck, log = log)
      output = output + error
      if 'Software Generation Utilities' in output:
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isAIXAR(ar, log):
    '''Returns true AR is AIX'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(ar + ' -V',checkCommand = noCheck, log = log)
      output = output + error
      if output.find('[-X{32|64|32_64|d64|any}]') >= 0:
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isUname(log):
    global isLinux_value,isCygwin_value,isSolaris_value,isDarwin_value,isDarwinCatalina_value,isFreeBSD_value,isUname_value
    isUname_value = True
    (output, error, status) = config.base.Configure.executeShellCommand('uname -s', log = log)
    if not status:
      output = output.lower().strip()
      if output.find('linux') >= 0:
        if log: log.write('Detected Linux OS')
        isLinux_value = True
        return
      if output.find('cygwin') >= 0:
        if log: log.write('Detected Cygwin')
        isCygwin_value = True
        return
      if output.find('sunos') >= 0:
        if log: log.write('Detected Solaris')
        isSolaris_value = True
        return
      if output.find('darwin') >= 0:
        if log: log.write('Detected Darwin')
        isDarwin_value = True
        import platform
        try:
          v = tuple([int(a) for a in platform.mac_ver()[0].split('.')])
          if v >= (10,15,0):
            if log: log.write('Detected Darwin/macOS Catalina OS\n')
            isDarwinCatalina_value = True
        except:
          if log: log.write('macOS version detecton failed!\n')
          pass
      if output.find('freebsd') >= 0:
        if log: log.write('Detected FreeBSD')
        isFreeBSD_value = True
        return

  @staticmethod
  def isLinux(log):
    '''Returns true if system is linux'''
    global isUname_value,isLinux_value
    if not isUname_value: config.setCompilers.Configure.isUname(log)
    return isLinux_value

  @staticmethod
  def isCygwin(log):
    '''Returns true if system is Cygwin'''
    global isUname_value,isCygwin_value
    if not isUname_value: config.setCompilers.Configure.isUname(log)
    return isCygwin_value

  @staticmethod
  def isSolaris(log):
    '''Returns true if system is Solaris'''
    global isUname_value,sSolaris_value
    if not isUname_value: config.setCompilers.Configure.isUname(log)
    return isSolaris_value

  @staticmethod
  def isDarwin(log):
    '''Returns true if system is Dwarwin'''
    global isUname_value,sDarwin_value
    if not isUname_value: config.setCompilers.Configure.isUname(log)
    return isDarwin_value

  @staticmethod
  def isDarwinCatalina(log):
    '''Returns true if system is Dwarwin Catalina'''
    global isUname_value,isDarwinCatalina_value
    if not isUname_value: config.setCompilers.Configure.isUname(log)
    return isDarwinCatalina_value

  @staticmethod
  def isFreeBSD(log):
    '''Returns true if system is FreeBSD'''
    global isUname_value,isFreeBSD_value
    if not isUname_value: config.setCompilers.Configure.isUname(log)
    return isFreeBSD_value

  @staticmethod
  def isARM(log):
    '''Returns true if system is processor-type is ARM'''
    global isARM_value
    if isARM_value == -1:
       (output, error, status) = config.base.Configure.executeShellCommand('uname -p', log = log)
       if not status and (output.lower().strip() == 'arm'):
         if log: log.write('Detected ARM processor\n\n')
         isARM_value = True
       else:
         isARM_value = False
    return isARM_value

  @staticmethod
  def addLdPath(path):
    if 'LD_LIBRARY_PATH' in os.environ:
      ldPath=os.environ['LD_LIBRARY_PATH']
    else:
      ldPath=''
    if ldPath == '': ldPath = path
    else: ldPath += ':' + path
    os.environ['LD_LIBRARY_PATH'] = ldPath
    return

  def useMPICompilers(self):
    if ('with-cc' in self.argDB and self.argDB['with-cc'] != '0') or 'CC' in self.argDB:
      return 0
    if ('with-cxx' in self.argDB and self.argDB['with-cxx'] != '0') or 'CXX' in self.argDB:
      return 0
    if ('with-fc' in self.argDB and self.argDB['with-fc'] != '0') or 'FC' in self.argDB:
      return 0
    if self.argDB['download-mpich'] or self.argDB['download-openmpi']:
      return 0
    if 'with-mpi-include' in self.argDB and self.argDB['with-mpi-include']:
      return 0;
    if 'with-mpi' in self.argDB and self.argDB['with-mpi'] and self.argDB['with-mpi-compilers']:
      return 1
    return 0

  def checkInitialFlags(self):
    '''Initialize the compiler and linker flags'''
    for language in ['C', 'CUDA', 'HIP', 'SYCL', 'Cxx', 'FC']:
      self.pushLanguage(language)
      for flagsArg in [config.base.Configure.getCompilerFlagsName(language), config.base.Configure.getCompilerFlagsName(language, 1), config.base.Configure.getLinkerFlagsName(language)]:
        if flagsArg in self.argDB: setattr(self, flagsArg, self.argDB[flagsArg])
        else: setattr(self, flagsArg, '')
        self.logPrint('Initialized '+flagsArg+' to '+str(getattr(self, flagsArg)))
      self.popLanguage()
    for flagsArg in ['CPPFLAGS', 'FPPFLAGS', 'CUDAPPFLAGS', 'CXXPPFLAGS', 'HIPPPFLAGS', 'SYCLPPFLAGS']:
      if flagsArg in self.argDB: setattr(self, flagsArg, self.argDB[flagsArg])
      else: setattr(self, flagsArg, '')
      self.logPrint('Initialized '+flagsArg+' to '+str(getattr(self, flagsArg)))
    # SYCLC_LINKER_FLAGS is init'ed above in the "for language" loop.
    # FIXME: these linker flags are init'ed as a list, while others are init'ed as a string. Need to make them consistent.
    for flagsArg in ['CC_LINKER_FLAGS', 'CXX_LINKER_FLAGS', 'FC_LINKER_FLAGS', 'CUDAC_LINKER_FLAGS', 'HIPC_LINKER_FLAGS', 'sharedLibraryFlags', 'dynamicLibraryFlags']:
      if isinstance(self.argDB[flagsArg],str): val = [self.argDB[flagsArg]]
      else: val = self.argDB[flagsArg]
      setattr(self, flagsArg, val)
      self.logPrint('Initialized '+flagsArg+' to '+str(getattr(self, flagsArg)))
    if 'LIBS' in self.argDB:
      self.LIBS = self.argDB['LIBS']
    else:
      self.LIBS = ''
    return

  def checkDeviceHostCompiler(self,language):
    """Set the host compiler (HC) of the device compiler (DC) to the HC unless the DC already explicitly sets its HC. This may be needed if the default HC used by the DC is ancient and PETSc uses a different HC (e.g., through --with-cxx=...)."""
    if language.upper() == 'CUDA':
      setHostFlag = '-ccbin'
    else:
      raise NotImplementedError
    with self.Language(language):
      if setHostFlag in self.getCompilerFlags():
        # don't want to override this if it is already set
        return
    if not hasattr(self,'CXX'):
        return
    compilerName = self.getCompiler(lang='Cxx')
    hostCCFlag   = '{shf} {cc}'.format(shf=setHostFlag,cc=compilerName)
    with self.Language(language):
      self.logPrint(' '.join(('checkDeviceHostCompiler: checking',language,'accepts host compiler',compilerName)))
      try:
        self.addCompilerFlag(hostCCFlag)
      except RuntimeError:
        pass
    return

  def checkCxxDialect(self, language, isGNUish=False):
    """Determine the CXX dialect supported by the compiler (language) [and corresponding compiler
    option - if any].

    isGNUish indicates if the compiler is gnu compliant (i.e. clang).
    -with-<lang>-dialect can take options:
      auto: use highest supported dialect configure can determine
      [[c|gnu][xx|++]]23: not yet supported
      [[c|gnu][xx|++]]20: gnu++20 or c++20
      [[c|gnu][xx|++]]17: gnu++17 or c++17
      [[c|gnu][xx|++]]14: gnu++14 or c++14
      [[c|gnu][xx|++]]11: gnu++11 or c++11
      0: disable CxxDialect check and use compiler default

    On return this function sets the following values:
    - if needed, appends the relevant CXX dialect flag to <lang> compiler flags
    - self.cxxDialectRange = (minSupportedDialect,maxSupportedDialect) (e.g. ('c++11','c++14'))
    - self.addDefine('HAVE_{LANG}_DIALECT_CXX{DIALECT_NUM}',1) for every supported dialect
    - self.lang+'dialect' = 'c++'+maxDialectNumber (e.g. 'c++14') but ONLY if the user
      specifically requests a dialect version, otherwise this is not set

    Raises a config.base.ConfigureSetupError if:
    - The user has set both the --with-dialect=[...] configure options and -std=[...] in their
      compiler flags
    - The combination of specifically requested packages cannot all be compiled with the same flag
    - An unknown C++ dialect is provided

    The config.base.ConfigureSetupErrors are NOT meant to be caught, as they are fatal errors
    on part of the user

    Raises a RuntimeError (which may be caught) if:
    - The compiler does not support at minimum -std=c++11
    """
    from config.base import ConfigureSetupError
    import textwrap

    def includes11():
      return textwrap.dedent(
        """
        // c++11 includes
        #include <memory>
        #include <random>
        #include <complex>
        #include <iostream>
        #include <algorithm>

        template<class T> void ignore(const T&) { } // silence unused variable warnings
        class valClass
        {
        public:
          int i;
          valClass() { i = 3; }
          valClass(int x) : i(x) { }
        };

        class MoveSemantics
        {
          std::unique_ptr<valClass> _member;

        public:
          MoveSemantics(int val = 4) : _member(new valClass(val)) { }
          MoveSemantics& operator=(MoveSemantics &&other) noexcept = default;
        };

        template<typename T> constexpr T Cubed( T x ) { return x*x*x; }
        auto trailing(int x) -> int { return x+2; }
        enum class Shapes : int {SQUARE,CIRCLE};
        template<class ... Types> struct Tuple { };
        using PetscErrorCode = int;
        """
      )

    def body11():
      return textwrap.dedent(
        """
        // c++11 body
        valClass cls = valClass(); // value initialization
        int i = cls.i;             // i is not declared const
        const int& rci = i;        // but rci is
        const_cast<int&>(rci) = 4;

        constexpr int big_value = 1234;
        decltype(big_value) ierr = big_value;
        auto ret = trailing(ierr);
        MoveSemantics bob;
        MoveSemantics alice;
        alice = std::move(bob);ignore(alice);
        Tuple<> t0;ignore(t0);
        Tuple<long> t1;ignore(t1);
        Tuple<int,float> t2;ignore(t2);
        std::random_device rd;
        std::mt19937 mt(rd());
        std::normal_distribution<double> dist(0,1);
        const double x = dist(mt);
        std::cout << x << ret << std::endl;
        std::vector<std::unique_ptr<double>> vector;
        std::sort(vector.begin(), vector.end(), [](std::unique_ptr<double> &a, std::unique_ptr<double> &b) { return *a < *b; });
        """
      )

    def includes14():
      return '\n'.join((includes11(),textwrap.dedent(
        """
        // c++14 includes
        #include <type_traits>

        template<class T> constexpr T pi = T(3.1415926535897932385L);  // variable template
        """
        )))

    def body14():
      return '\n'.join((body11(),textwrap.dedent(
        """
        // c++14 body
        auto ptr = std::make_unique<int>();
        *ptr = 1;
        std::cout << pi<double> << std::endl;
        constexpr const std::complex<double> const_i(0.0,1.0);
        auto lambda = [](auto x, auto y) { return x + y; };
        std::cout << lambda(3,4) << std::real(const_i) << std::endl;
        """
      )))

    def includes17():
      return '\n'.join((includes14(),textwrap.dedent(
        """
        // c++17 includes
        #include <string_view>
        #include <any>
        #include <optional>
        #include <variant>
        #include <tuple>
        #include <new>

        std::align_val_t dummy;
        [[nodiscard]] int nodiscardFunc() { return 0; }
        struct S2
        {
          // static inline member variables since c++17
          static inline int var = 8675309;
          void f(int i);
        };
        void S2::f(int i)
        {
          // until c++17: Error: invalid syntax
          // since c++17: OK: captures the enclosing S2 by copy
          auto lmbd = [=, *this] { std::cout << i << " " << this->var << std::endl; };
          lmbd();
        }
        std::tuple<double, int, char> foobar()
        {
          return {3.8, 0, 'x'};
        }
        """
      )))

    def body17():
      return '\n'.join((body14(),textwrap.dedent(
        """
        // c++17 body
        std::variant<int,float> v,w;
        v = 42;               // v contains int
        int ivar = std::get<int>(v);
        w = std::get<0>(v);   // same effect as the previous line
        w = v;                // same effect as the previous line
        S2 foo;
        foo.f(ivar);
        if constexpr (std::is_arithmetic_v<int>) std::cout << "c++17" << std::endl;
        typedef std::integral_constant<Shapes,Shapes::SQUARE> squareShape;
        // static_assert with no message since c++17
        static_assert(std::is_same_v<squareShape,squareShape>);
        auto val = nodiscardFunc();ignore(val);
        // structured binding
        const auto [ab, cd, ef] = foobar();
        """
      )))

    def includes20():
      return '\n'.join((includes17(),textwrap.dedent(
        """
        // c++20 includes
        #include <compare>
        #include <concepts>

        consteval int sqr_cpp20(int n)
        {
          return n*n;
        }
        constexpr auto r = sqr_cpp20(10);
        static_assert(r == 100);

        const char *g_cpp20() { return "dynamic initialization"; }
        constexpr const char *f_cpp20(bool p) { return p ? "constant initializer" : g_cpp20(); }
        constinit const char *cinit_c = f_cpp20(true); // OK

        // Declaration of the concept "Hashable", which is satisfied by any type 'T'
        // such that for values 'a' of type 'T', the expression std::hash<T>{}(a)
        // compiles and its result is convertible to std::size_t
        template <typename T>
        concept Hashable = requires(T a)
        {
          { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
        };

        struct meow {};

        // Constrained C++20 function template:
        template <Hashable T>
        void f_concept(T) {}

        void abbrev_f1(auto); // same as template<class T> void abbrev_f1(T)
        void abbrev_f4(const std::destructible auto*, std::floating_point auto&); // same as template<C3 T, C4 U> void abbrev_f4(const T*, U&);

        template<>
        void abbrev_f4<int>(const int*, const double&); // specialization of abbrev_f4<int, const double> (since C++20)
        """
      )))

    def body20():
      return '\n'.join((body17(),textwrap.dedent(
        """
        // c++20 body
        ignore(cinit_c);

        using std::operator""s;
        f_concept("abc"s);
        """
      )))

    isGNUish     = bool(isGNUish)
    lang,LANG    = language.lower(),language.upper()
    compiler     = self.getCompiler(lang=language)
    if self.isMSVC(compiler, self.log):
      stdflag_base = '-std:'
    elif self.isWindows(compiler, self.log) and self.isIntel(compiler, self.log):
      stdflag_base = '-Qstd='
    else:
      stdflag_base = '-std='
    DialectFlags = namedtuple('DialectFlags',['standard','gnu'])
    BaseFlags    = DialectFlags(standard=stdflag_base+'c++',gnu=stdflag_base+'gnu++')
    self.logPrint('checkCxxDialect: checking C++ dialect version for language "{lang}" using compiler "{compiler}"'.format(lang=LANG,compiler=compiler))
    self.logPrint('checkCxxDialect: PETSc believes compiler ({compiler}) {isgnuish} gnu-ish'.format(compiler=compiler,isgnuish='IS' if isGNUish else 'is NOT'))

    # if we have done this before the flag may have been inserted (by us) into the
    # compiler flags, so we shouldn't yell at the user for having it in there, nor should
    # we treat it as explicitly being set. If we have the attribute, it is either True or
    # False
    setPreviouslyAttrName   = lang+'dialect_set_explicitly__'
    previouslySetExplicitly = getattr(self,setPreviouslyAttrName,None)
    assert previouslySetExplicitly in (True,False,None)
    processedBefore = previouslySetExplicitly is not None
    self.logPrint('checkCxxDialect: PETSc believes that we {pbefore} processed {compiler} before'.format(pbefore='HAVE' if processedBefore else 'have NOT',compiler=compiler))

    # configure value
    useFlag         = True
    configureArg    = lang.join(['with-','-dialect'])
    withLangDialect = self.argDB.get(configureArg).upper().replace('X','+')
    if withLangDialect in ('','0','NONE'):
      self.logPrint(
        'checkCxxDialect: user has requested NO cxx dialect, we\'ll check but not add the flag'
      )
      withLangDialect = 'NONE'
      useFlag         = False # we still do the checks, just not add the flag in the end
    self.logPrint('checkCxxDialect: configure option after sanitization: --{opt}={val}'.format(opt=configureArg,val=withLangDialect))

    # check the configure argument
    if withLangDialect.startswith('GNU'):
      allowedBaseFlags = [BaseFlags.gnu]
    elif withLangDialect.startswith('C++'):
      allowedBaseFlags = [BaseFlags.standard]
    elif withLangDialect == 'NONE':
      allowedBaseFlags = ['(NO FLAG)']
    else:
      # if we are here withLangDialect is either AUTO or e.g. 14
      allowedBaseFlags = [BaseFlags.gnu,BaseFlags.standard] if isGNUish else [BaseFlags.standard]

    Dialect  = namedtuple('Dialect',['num','includes','body'])
    dialects = (
      Dialect(num='11',includes=includes11(),body=body11()),
      Dialect(num='14',includes=includes14(),body=body14()),
      Dialect(num='17',includes=includes17(),body=body17()),
      Dialect(num='20',includes=includes20(),body=body20()),
      Dialect(num='23',includes=includes20(),body=body20()) # no c++23 checks yet
    )

    # search compiler flags to see if user has set the c++ standard from there
    with self.Language(language):
      allFlags = tuple(self.getCompilerFlags().strip().split())
    langDialectFromFlags = tuple(f for f in allFlags for flg in BaseFlags if f.startswith(flg))
    if len(langDialectFromFlags):
      sanitized = langDialectFromFlags[-1].lower().replace(stdflag_base,'')
      if not processedBefore:
        # check that we didn't set the compiler flag ourselves before we yell at the user
        if withLangDialect != 'AUTO':
          # user has set both flags
          errorMessage = 'Competing or duplicate C++ dialect flags, have specified {flagdialect} in compiler ({compiler}) flags and used configure option {opt}'.format(flagdialect=langDialectFromFlags,compiler=compiler,opt='--'+configureArg+'='+withLangDialect.lower())
          raise ConfigureSetupError(errorMessage)
        mess = 'Explicitly setting C++ dialect in compiler flags may not be optimal. Use ./configure --{opt}={sanitized} if you really want to use that value, otherwise remove {flag} from compiler flags and omit --{opt}=[...] from configure to have PETSc automatically detect the most appropriate flag for you'.format(opt=configureArg,sanitized=sanitized,flag=langDialectFromFlags[-1])
        self.logPrintWarning(mess)

      # the user has already set the flag in their options, no need to set it a second time
      useFlag          = False
      # set the dialect to whatever was in the users compiler flags
      withLangDialect  = sanitized
      # if we have processed before, then the flags will be the ones we set, so it's best
      # to just keep the allowedBaseFlags general
      if not processedBefore:
        allowedBaseFlags = [
          BaseFlags.gnu if withLangDialect.startswith('gnu') else BaseFlags.standard
        ]

    # delete any previous defines (in case we are doing this again)
    for dlct in dialects:
      self.delDefine('HAVE_{lang}_DIALECT_CXX{ver}'.format(lang=LANG,ver=dlct.num))

    if withLangDialect in {'AUTO','NONE'}:
      # see top of file
      dialectNumStr = default_cxx_dialect_ranges()[1]
      explicit      = withLangDialect == 'NONE' # NONE is explicit but AUTO is not
    else:
      # we can stop shouting now
      dialectNumStr = withLangDialect = withLangDialect.lower()
      # if we have done this before, then previouslySetExplicitly holds the previous
      # explicit value
      explicit      = previouslySetExplicitly if processedBefore else True
      max_sup_ver   = default_cxx_dialect_ranges()[1].replace('c++','')
      max_unsup_ver = str(int(max_sup_ver) + 3)
      if withLangDialect.endswith(max_unsup_ver):
        mess = 'C++{unsup_ver} is not yet fully supported, PETSc only tests up to C++{maxver}. Remove -std=[...] from compiler flags and/or omit --{opt}=[...] from configure to have PETSc automatically detect the most appropriate flag for you'.format(unsup_ver=max_unsup_ver,maxver=max_sup_ver,opt=configureArg)
        self.logPrintWarning(mess)

    minDialect,maxDialect = 0,-1
    for i,dialect in enumerate(dialects):
      if dialectNumStr.endswith(dialect.num):
        maxDialect = i
        break

    if maxDialect == -1:
      try:
        ver = int(withLangDialect[-2:])
      except ValueError:
        ver = 9e9
      minver = int(dialects[0].num)
      if ver in {89, 98} or ver < minver:
        mess = 'PETSc requires at least C++{} when using {}'.format(minver, language.replace('x', '+'))
      else:
        mess = 'Unknown C++ dialect: {}'.format(withLangDialect)
      if explicit:
        mess += ' (you have explicitly requested --{}={}). Remove this flag and let configure choose the most appropriate flag for you.'.format(configureArg, withLangDialect)
        # If the user explicitly requested the dialect throw CSE (which is NOT meant to be
        # caught) as this indicates a user error
        raise ConfigureSetupError(mess)
      raise RuntimeError(mess)
    self.logPrint('checkCxxDialect: dialect {dlct} has been {expl} selected for {lang}'.format(dlct=withLangDialect,expl='EXPLICITLY' if explicit else 'NOT explicitly',lang=LANG))

    def checkPackageRange(packageRanges,kind,dialectIdx):
      if kind == 'upper':
        boundFunction   = min
        compareFunction = lambda x,y: x[-2:] > y[-2:]
      elif kind == 'lower':
        boundFunction   = max
        compareFunction = lambda x,y: x == 'NONE' or x[-2:] < y[-2:]
      else:
        raise ValueError('unknown bound type',kind)

      # Check that we have a sane upper bound on the dialect
      if len(packageRanges.keys()):
        packageBound = boundFunction(packageRanges.keys()).lower()
        startDialect = withLangDialect if explicit else dialects[dialectIdx].num
        if compareFunction(startDialect,packageBound):
          packageBlame = '\n'.join('\t- '+s for s in packageRanges[packageBound])
          # if using NONE startDialect will be highest possible dialect
          if explicit and startDialect != 'NONE':
            # user asked for a dialect, they'll probably want to know why it doesn't work
            errorMessage = '\n'.join((
              'Explicitly requested {lang} dialect {dlct} but package(s):',
              packageBlame.replace('\t',''),
              'Has {kind} bound of -std={packdlct}'
            )).format(lang=LANG,dlct=withLangDialect,kind=kind,packdlct=packageBound)
            raise ConfigureSetupError(errorMessage)
          # if not explicit, we can just silently log the discrepancy instead
          self.logPrint('\n'.join((
            'checkCxxDialect: had {lang} dialect {dlct} as {kind} bound but package(s):',
            packageBlame,
            '\tHas {kind} bound of -std={packdlct}, using package requirement -std={packdlct}'
          )).format(lang=LANG,dlct=startDialect,kind=kind,packdlct=packageBound))
          try:
            dialectIdx = [i for i,d in enumerate(dialects) if packageBound.endswith(d.num)][0]
          except IndexError:
            mess = 'Could not find a dialect number that matches the package bounds: {}'.format(
              packageRanges
            )
            raise ConfigureSetupError(mess)
      return dialectIdx


    maxDialect = checkPackageRange(self.cxxDialectPackageRanges[1],'upper',maxDialect)
    minDialect = checkPackageRange(self.cxxDialectPackageRanges[0],'lower',minDialect)

    # if the user asks for a particular version we should pin that version
    if withLangDialect not in ('NONE','AUTO') and explicit:
      minDialect = maxDialect

    # compile a list of all the flags we will test in descending order, for example
    # -std=gnu++17
    # -std=c++17
    # -std=gnu++14
    # ...
    flagPool = [(''.join((b,d.num)),d) for d in reversed(dialects[minDialect:maxDialect+1]) for b in allowedBaseFlags]

    self.logPrint(
      '\n'.join(['checkCxxDialect: Have potential flag pool:']+['\t   - '+f for f,_ in flagPool])
    )
    assert len(flagPool)
    with self.Language(language):
      for index,(flag,dlct) in enumerate(flagPool):
        self.logPrint(' '.join(('checkCxxDialect: checking CXX',dlct.num,'for',lang,'with',flag)))
        # test with flag
        try:
          if useFlag:
            # needs compilerOnly = True as we need to keep the flag out of the linker flags
            self.addCompilerFlag(flag,includes=dlct.includes,body=dlct.body,compilerOnly=True)
          elif not self.checkCompile(includes=dlct.includes,body=dlct.body):
            raise RuntimeError # to mimic addCompilerFlag
        except RuntimeError:
          # failure, flag is discarded, but first check we haven't run out of flags
          if index == len(flagPool)-1:
            # compiler does not support the minimum required c++ dialect
            base_mess = '\n'.join((
              '{lang} compiler ({compiler}) appears non-compliant with C++{ver} or didn\'t accept:',
              '\n'.join(
                '- '+(f[:-2] if f.startswith('(NO FLAG)') else f) for f,_ in flagPool[:index+1]
              ),
              '' # for extra newline at the end
            ))
            if withLangDialect in ('NONE','AUTO'):
              packDialects = self.cxxDialectPackageRanges[0]
              if packDialects.keys():
                # it's a packages fault we can't try the next dialect
                minPackDialect = max(packDialects.keys())
                base_mess      = '\n'.join((
                  'Using {lang} dialect C++{ver} as lower bound due to package(s):',
                  '\n'.join('- '+s for s in packDialects[minPackDialect]),
                  ' '.join(('But',base_mess))
                ))
                dialectNum = minPackDialect[-2:]
                explicit   = True
              else:
                assert flag.endswith(dialects[0].num)
                # it's the compilers fault we can't try the next dialect
                dialectNum = dialects[0].num
            else:
              # if nothing else then it's because the user requested a particular version
              dialectNum = dialectNumStr
              base_mess  = '\n'.join((
                base_mess,
                'Note, you have explicitly requested --{}={}. If you do not need C++{ver}, then remove this flag and let configure choose the most appropriate flag for you.'
                '\nIf you DO need it, then (assuming your compiler isn\'t just old) try consulting your compilers user manual. There may be other flags (e.g. \'--gcc-toolchain\') you must pass to enable C++{ver}'.format(configureArg, withLangDialect, ver='{ver}')
              ))
            if dialectNum.isdigit():
              ver = dialectNum
            else:
              ver = dialectNum.casefold().replace('c++', '').replace('gnu++', '')
            mess = base_mess.format(lang=language.replace('x','+'),compiler=compiler,ver=ver)
            if explicit:
              # if the user explicitly set the version, then this is a hard error
              raise ConfigureSetupError(mess)
            raise RuntimeError(mess)
        else:
          # success
          self.cxxDialectRange[language] = ('c++'+dialects[minDialect].num,'c++'+dlct.num)
          if not useFlag:
            compilerFlags = self.getCompilerFlags()
            if compilerFlags.count(flag) > 1:
              errorMessage = '\n'.join((
                'We said we wouldn\'t add the flag yet the flag has been mysteriously added!!:',
                compilerFlags
              ))
              raise ConfigureSetupError(errorMessage)
          self.logPrint('checkCxxDialect: success using {flag} for {lang} dialect C++{ver}, set new cxxDialectRange: {drange}'.format(flag=flag,lang=language,ver=dlct.num,drange=self.cxxDialectRange[language]))
          break # flagPool loop

    # this loop will also set maxDialect for the setattr below
    for maxDialect,dlct in enumerate(dialects):
      if dlct.num > flag[-2:]:
        break
      self.addDefine('HAVE_{lang}_DIALECT_CXX{ver}'.format(lang=LANG,ver=dlct.num),1)

    if explicit:
      # if we don't use the flag we shouldn't set this attr because its existence implies
      # a particular dialect is *chosen*
      setattr(self,lang+'dialect','c++'+dialects[maxDialect-1].num)
    setattr(self,setPreviouslyAttrName,explicit)
    return

  def checkCompiler(self, language, linkLanguage=None,includes = '', body = '', cleanup = 1, codeBegin = None, codeEnd = None):
    """Check that the given compiler is functional, and if not raise an exception"""
    with self.Language(language):
      compiler = self.getCompiler()
      if not self.checkCompile(includes=includes,body=body,cleanup=cleanup,codeBegin=codeBegin,codeEnd=codeEnd):
        msg = 'Cannot compile {} with {}.'.format(language,compiler)
        raise RuntimeError(msg)

      if language.upper() in {'CUDA','HIP','SYCL'}:
        # do not check CUDA/HIP/SYCL linkers since they are never used (assumed for now)
        return
      if not self.checkLink(linkLanguage=linkLanguage,includes=includes,body=body):
        msg = 'Cannot compile/link {} with {}.'.format(language,compiler)
        msg = '\nIf the above linker messages do not indicate failure of the compiler you can rerun with the option --ignoreLinkOutput=1'
        raise RuntimeError(msg)
      oldlibs     = self.LIBS
      compilerObj = self.framework.getCompilerObject(linkLanguage if linkLanguage else language)
      if not hasattr(compilerObj,'linkerrorcodecheck'):
        self.LIBS += ' -lpetsc-ufod4vtr9mqHvKIQiVAm'
        if self.checkLink(linkLanguage=linkLanguage):
          self.LIBS = oldlibs
          msg = '\n'.join((
            '{lang} compiler {cmp} is broken! It is returning no errors for a failed link! Either',
            '1) switch to another compiler suite',
            '2) report this entire error message to your compiler/linker suite vendor'
          )).format(lang=language,cmp=compiler)
          raise RuntimeError(msg)
        self.LIBS = oldlibs
        compilerObj.linkerrorcodecheck = 1
      if not self.argDB['with-batch']:
        if not self.checkRun(linkLanguage=linkLanguage):
          msg = '\n'.join((
            'Cannot run executables created with {language}. If this machine uses a batch system ',
            'to submit jobs you will need to configure using ./configure with the additional option  --with-batch. ',
            'Otherwise there is problem with the compilers. Can you compile and run code with your compiler \'{compiler}\'?'
          )).format(language=language,compiler=compiler)
          if self.isIntel(compiler,self.log):
            msg = '\n'.join((msg,'See https://petsc.org/release/faq/#error-libimf'))
          raise OSError(msg) # why OSError?? it isn't caught anywhere in here?
    return


  def crayCrossCompiler(self,compiler):
    import script
    '''For Cray Intel KNL systems returns the underlying compiler line used by the wrapper compiler if is for KNL systems'''
    '''This removes all the KNL specific options allowing the generated binary to run on the front-end'''
    '''This is needed by some build systems include HDF5 that insist on running compiled programs during the configure and'''
    '''make process. This does not work for the Cray compiler module, only intel and gcc'''

    (output,error,status) = self.executeShellCommand(compiler+' -craype-verbose',checkCommand = script.Script.passCheckCommand,log=self.log)
    output = output.split()
    if output[0].strip().startswith('driver'): return ''
    newoutput = [output[0]]
    cross = 0
    for i in output[1:-1]:
      if i.find('mic') > -1 or i.find('knl') > -1 or i.find('KNL') > -1:
        cross = 1
        continue
      if i.startswith('-L') or i.startswith('-l') or i.startswith('-Wl'):
        continue
      newoutput.append(i)
    if cross:
      return ' '.join(newoutput)
    return ''

  def crayCrossLIBS(self,compiler):
    import script
    '''For Cray Intel KNL systems returns the underlying linker options used by the wrapper compiler if is for KNL systems'''
    (output,error,status) = self.executeShellCommand(compiler+' -craype-verbose',checkCommand = script.Script.passCheckCommand,log=self.log)
    output = output.split()
    newoutput = []
    cross = 0
    for i in output[1:-1]:
      if i.find('mic') > -1 or i.find('knl') > -1 or i.find('KNL') > -1:
        cross = 1
        continue
      if i.find('darshan') > -1:
        cross = 1
        continue
      if i.find('static') > -1:
        continue
      if i.startswith('-I') or i.startswith('-D'):
        continue
      # the math libraries are not needed by external packages and cause errors in HDF5 with libgfortran.so.4 => not found
      if i.startswith('-lsci_gnu'):
        continue
      newoutput.append(i)
    if cross:
      return ' '.join(newoutput)
    return ''


  def generateCCompilerGuesses(self):
    '''Determine the C compiler '''
    if hasattr(self, 'CC'):
      yield self.CC
      if self.argDB['download-mpich']: mesg ='with downloaded MPICH'
      elif self.argDB['download-openmpi']: mesg ='with downloaded Open MPI'
      else: mesg = ''
      raise RuntimeError('Error '+mesg+': '+self.mesg)
    elif 'with-cc' in self.argDB:
      yield self.argDB['with-cc']
      raise RuntimeError('C compiler you provided with -with-cc='+self.argDB['with-cc']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif 'CC' in self.argDB:
      yield self.argDB['CC']
      raise RuntimeError('C compiler you provided with -CC='+self.argDB['CC']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif self.useMPICompilers() and 'with-mpi-dir' in self.argDB and os.path.isdir(os.path.join(self.argDB['with-mpi-dir'], 'bin')):
      self.usedMPICompilers = 1
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpincc')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpiicc')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpicc')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpcc')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'hcc')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpcc_r')
      self.usedMPICompilers = 0
      raise RuntimeError('MPI compiler wrappers in '+self.argDB['with-mpi-dir']+'/bin cannot be found or do not work. See https://petsc.org/release/faq/#invalid-mpi-compilers')
    else:
      if self.useMPICompilers() and 'with-mpi-dir' in self.argDB:
      # if it gets here these means that self.argDB['with-mpi-dir']/bin does not exist so we should not search for MPI compilers
      # that is we are turning off the self.useMPICompilers()
        self.logPrintWarning(os.path.join(self.argDB['with-mpi-dir'], 'bin')+ ' dir does not exist! Skipping check for MPI compilers due to potentially incorrect --with-mpi-dir option. Suggest using --with-cc=/path/to/mpicc option instead')

        self.argDB['with-mpi-compilers'] = 0
      if self.useMPICompilers():
        self.usedMPICompilers = 1
        cray = os.getenv('CRAYPE_DIR')
        if cray:
          cross_cc = self.crayCrossCompiler('cc')
          if cross_cc:
            self.cross_cc = cross_cc
            self.log.write('Cray system using C cross compiler:'+cross_cc+'\n')
            self.cross_LIBS = self.crayCrossLIBS('cc')
            self.log.write('Cray system using C cross LIBS:'+self.cross_LIBS+'\n')
          yield 'cc'
          if cross_cc:
            delattr(self, 'cross_cc')
            delattr(self, 'cross_LIBS')
        yield 'mpincc'
        yield 'mpicc'
        yield 'mpiicc'
        yield 'mpcc_r'
        yield 'mpcc'
        yield 'mpxlc'
        yield 'hcc'
        self.usedMPICompilers = 0
      yield 'ncc'
      yield 'gcc'
      yield 'clang'
      yield 'icc'
      yield 'cc'
      yield 'xlc'
      path = os.path.join(os.getcwd(),'lib','petsc','win32fe','bin')
      yield os.path.join(path,'win32fe_icl')
      yield os.path.join(path,'win32fe_cl')
      yield 'pgcc'
    return

  def showMPIWrapper(self,compiler):
    if os.path.basename(compiler).startswith('mpi'):
      self.logPrint(' MPI compiler wrapper '+compiler+' failed to compile')
      try:
        output = self.executeShellCommand(compiler + ' -show', log = self.log)[0]
      except RuntimeError:
        self.logPrint('-show option failed for MPI compiler wrapper '+compiler)
    self.logPrint(' MPI compiler wrapper '+compiler+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI.')

  def checkCCompiler(self):
    import re
    '''Locate a functional C compiler'''
    if 'with-cc' in self.argDB and self.argDB['with-cc'] == '0':
      raise RuntimeError('A functional C compiler is necessary for configure, cannot use --with-cc=0')
    self.mesg = ''
    for compiler in self.generateCCompilerGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'CC'):
          self.checkCompiler('C')
          break
      except RuntimeError as e:
        self.mesg = str(e)
        self.logPrint('Error testing C compiler: '+str(e))
        self.showMPIWrapper(compiler)
        self.delMakeMacro('CC')
        del self.CC
    if not hasattr(self, 'CC'):
      raise RuntimeError('Could not locate a functional C compiler')
    try:
      (output,error,status) = self.executeShellCommand(self.CC+' --version', log = self.log)
    except:
      pass
    else:
      if self.isDarwin(self.log) and self.isARM(self.log) and output.find('x86_64-apple-darwin') > -1:
        raise RuntimeError('Running on a macOS ARM system but your compilers are configured for Intel processors\n' + output + '\n')

    (output, error, status) = config.base.Configure.executeShellCommand(self.CC+' -v | head -n 20', log = self.log)
    output = output + error
    if '(gcc version 4.8.5 compatibility)' in output or re.match('^Selected GCC installation:.*4.8.5$', output):
       self.logPrintWarning('Intel compiler being used with gcc 4.8.5 compatibility, failures may occur. Recommend having a newer gcc version in your path.')
    if os.path.basename(self.CC).startswith('mpi'):
       self.logPrint('Since MPI c compiler starts with mpi, force searches for other compilers to only look for MPI compilers\n')
       self.argDB['with-mpi-compilers'] = 1
    return

  def generateCPreprocessorGuesses(self):
    '''Determines the C preprocessor from CPP, then --with-cpp, then the C compiler'''
    if 'with-cpp' in self.argDB:
      yield self.argDB['with-cpp']
    elif 'CPP' in self.argDB:
      yield self.argDB['CPP']
    else:
      yield self.CC+' -E'
      yield self.CC+' --use cpp32'
    return

  def checkCPreprocessor(self):
    '''Locate a functional C preprocessor'''
    with self.Language('C'):
      for compiler in self.generateCPreprocessorGuesses():
        try:
          if self.getExecutable(compiler, resultName = 'CPP'):
            if not self.checkPreprocess('#include <stdlib.h>\n'):
              raise RuntimeError('Cannot preprocess C with '+self.CPP+'.')
            return
        except RuntimeError as e:
          self.logPrint(str(e))
    raise RuntimeError('Cannot find a C preprocessor')
    return


  def generateCUDACompilerGuesses(self):
    '''Determine the CUDA compiler using CUDAC, then --with-cudac
       - Any given category can be excluded'''
    if hasattr(self, 'CUDAC'):
      yield self.CUDAC
      raise RuntimeError('Error: '+self.mesg)
    elif 'with-cudac' in self.argDB:
      yield self.argDB['with-cudac']
      raise RuntimeError('CUDA compiler you provided with -with-cudac='+self.argDB['with-cudac']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif 'CUDAC' in self.argDB:
      yield self.argDB['CUDAC']
      raise RuntimeError('CUDA compiler you provided with -CUDAC='+self.argDB['CUDAC']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif 'with-cuda-dir' in self.argDB:
      nvccPath = os.path.join(self.argDB['with-cuda-dir'], 'bin','nvcc')
      yield nvccPath
    else:
      yield 'nvcc'
      yield os.path.join('/Developer','NVIDIA','CUDA-6.5','bin','nvcc')
      yield os.path.join('/usr','local','cuda','bin','nvcc')
      yield 'clang'
    return

  def checkCUDACompiler(self):
    '''Locate a functional CUDA compiler'''
    self.mesg = ''
    for compiler in self.generateCUDACompilerGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'CUDAC'):
          self.checkCompiler('CUDA')
          # Put version info into the log
          self.executeShellCommand(self.CUDAC+' --version', log = self.log)
          break
      except RuntimeError as e:
        self.mesg = str(e)
        self.logPrint('Error testing CUDA compiler: '+str(e))
        self.delMakeMacro('CUDAC')
        del self.CUDAC
    return

  def generateCUDAPreprocessorGuesses(self):
    '''Determines the CUDA preprocessor from --with-cudapp, then CUDAPP, then the CUDA compiler'''
    if 'with-cudacpp' in self.argDB:
      yield self.argDB['with-cudapp']
    elif 'CUDAPP' in self.argDB:
      yield self.argDB['CUDAPP']
    else:
      if hasattr(self, 'CUDAC'):
        yield self.CUDAC+' -E'
    return

  def checkCUDAPreprocessor(self):
    '''Locate a functional CUDA preprocessor'''
    with self.Language('CUDA'):
      for compiler in self.generateCUDAPreprocessorGuesses():
        try:
          if self.getExecutable(compiler, resultName = 'CUDAPP'):
            if not self.checkPreprocess('#include <stdlib.h>\n__global__ void testFunction() {return;};'):
              raise RuntimeError('Cannot preprocess CUDA with '+self.CUDAPP+'.')
            return
        except RuntimeError as e:
          self.logPrint(str(e))
    return


  def generateHIPCompilerGuesses(self):
    '''Determine the HIP compiler using HIPC, then --with-hipc
       - Any given category can be excluded'''
    if hasattr(self, 'HIPC'):
      yield self.HIPC
      raise RuntimeError('Error: '+self.mesg)
    elif 'with-hipc' in self.argDB:
      yield self.argDB['with-hipc']
      raise RuntimeError('HIPC compiler you provided with -with-hipc='+self.argDB['with-hipc']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif 'HIPC' in self.argDB:
      yield self.argDB['HIPC']
      raise RuntimeError('HIP compiler you provided with -HIPC='+self.argDB['HIPC']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif 'with-hip-dir' in self.argDB:
      hipPath = os.path.join(self.argDB['with-hip-dir'], 'bin','hipcc')
      yield hipPath
    else:
      yield 'hipcc'
      yield os.path.join('opt','rocm','bin','hipcc')
    return

  def checkHIPCompiler(self):
    '''Locate a functional HIP compiler'''
    self.mesg = 'in generateHIPCompilerGuesses'
    for compiler in self.generateHIPCompilerGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'HIPC'):
          self.checkCompiler('HIP')
          # Put version info into the log
          self.executeShellCommand(self.HIPC+' --version', log = self.log)
          break
      except RuntimeError as e:
        self.mesg = str(e)
        self.logPrint('Error testing HIP compiler: '+str(e))
        self.delMakeMacro('HIPC')
        del self.HIPC
    return

  def generateHIPPreprocessorGuesses(self):
    '''Determines the HIP preprocessor from --with-hippp, then HIPPP, then the HIP compiler'''
    if 'with-hipcpp' in self.argDB:
      yield self.argDB['with-hippp']
    elif 'HIPPP' in self.argDB:
      yield self.argDB['HIPPP']
    else:
      if hasattr(self, 'HIPC'):
        yield self.HIPC+' -E'
    return

  def checkHIPPreprocessor(self):
    '''Locate a functional HIP preprocessor'''
    with self.Language('HIP'):
      for compiler in self.generateHIPPreprocessorGuesses():
        try:
          if self.getExecutable(compiler, resultName = 'HIPPP'):
            if not self.checkPreprocess('#include <stdlib.h>\n__global__ void testFunction() {return;};'):
              raise RuntimeError('Cannot preprocess HIP with '+self.HIPPP+'.')
            return
        except RuntimeError as e:
          self.logPrint(str(e))
    return


  def generateSYCLCompilerGuesses(self):
    '''Determine the SYCL compiler using SYCLC, then --with-syclc
       - Any given category can be excluded'''
    if hasattr(self, 'SYCLC'):
      yield self.SYCLC
      raise RuntimeError('Error: '+self.mesg)
    elif 'with-syclc' in self.argDB:
      yield self.argDB['with-syclc']
      raise RuntimeError('SYCLC compiler you provided with -with-syclxx='+self.argDB['with-syclxx']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif 'SYCLC' in self.argDB:
      yield self.argDB['SYCLC']
      raise RuntimeError('SYCLC compiler you provided with -SYCLC='+self.argDB['SYCLC']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif 'with-sycl-dir' in self.argDB:
      syclPath = os.path.join(self.argDB['with-sycl-dir'], 'bin','dpcpp')
      yield syclPath
    return

  def checkSYCLCompiler(self):
    '''Locate a functional SYCL compiler'''
    self.mesg = 'in generateSYCLCompilerGuesses'
    for compiler in self.generateSYCLCompilerGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'SYCLC'):
          self.checkCompiler('SYCL')
          # Put version info into the log
          self.executeShellCommand(self.SYCLC+' --version', log = self.log)
          break
      except RuntimeError as e:
        self.mesg = str(e)
        self.delMakeMacro('SYCLC')
        del self.SYCLC
    return

  def generateSYCLPreprocessorGuesses(self):
    '''Determines the SYCL preprocessor from --with-syclpp, then SYCLPP, then the SYCL compiler'''
    if 'with-syclpp' in self.argDB:
      yield self.argDB['with-syclpp']
    elif 'SYCLPP' in self.argDB:
      yield self.argDB['SYCLPP']
    else:
      if hasattr(self, 'SYCLC'):
        yield self.SYCLC +' -E'
    return

  def checkSYCLPreprocessor(self):
    '''Locate a functional SYCL preprocessor'''
    with self.Language('SYCL'):
      for compiler in self.generateSYCLPreprocessorGuesses():
        try:
          if self.getExecutable(compiler, resultName = 'SYCLPP'):
            if not self.checkPreprocess('#include <sycl/sycl.hpp>\n void testFunction() {return;};'):
              raise RuntimeError('Cannot preprocess SYCL with '+self.SYCLPP+'.')
            return
        except RuntimeError as e:
          self.logPrint(str(e))
    return


  def generateCxxCompilerGuesses(self):
    '''Determine the Cxx compiler'''

    if hasattr(self, 'CXX'):
      yield self.CXX
      if self.argDB['download-mpich']: mesg ='with downloaded MPICH'
      elif self.argDB['download-openmpi']: mesg ='with downloaded Open MPI'
      else: mesg = ''
      raise RuntimeError('Error '+mesg+': '+self.mesg)
    elif 'with-c++' in self.argDB:
      raise RuntimeError('Keyword --with-c++ is WRONG, use --with-cxx')
    if 'with-CC' in self.argDB:
      raise RuntimeError('Keyword --with-CC is WRONG, use --with-cxx')

    if 'with-cxx' in self.argDB:
      if self.argDB['with-cxx'] == 'gcc': raise RuntimeError('Cannot use C compiler gcc as the C++ compiler passed in with --with-cxx')
      yield self.argDB['with-cxx']
      raise RuntimeError('C++ compiler you provided with -with-cxx='+self.argDB['with-cxx']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif 'CXX' in self.argDB:
      yield self.argDB['CXX']
      raise RuntimeError('C++ compiler you provided with -CXX='+self.argDB['CXX']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif self.usedMPICompilers and 'with-mpi-dir' in self.argDB and os.path.isdir(os.path.join(self.argDB['with-mpi-dir'], 'bin')):
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpinc++')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpiicpc')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpicxx')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'hcp')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpic++')
      if not Configure.isDarwin(self.log):
        yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpiCC')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpCC_r')
      raise RuntimeError('bin/<mpiCC,mpicxx,hcp,mpCC_r> you provided with -with-mpi-dir='+self.argDB['with-mpi-dir']+' cannot be found or does not work. See https://petsc.org/release/faq/#invalid-mpi-compilers')
    else:
      if self.usedMPICompilers:
        # TODO: Should only look for the MPI CXX compiler related to the found MPI C compiler
        cray = os.getenv('CRAYPE_DIR')
        if cray:
          cross_CC = self.crayCrossCompiler('CC')
          if cross_CC:
            self.cross_CC = cross_CC
            self.log.write('Cray system using C++ cross compiler:'+cross_CC+'\n')
          yield 'CC'
          if cross_CC: delattr(self, 'cross_CC')
        yield 'mpinc++'
        yield 'mpicxx'
        yield 'mpiicpc'
        yield 'mpCC_r'
        if not Configure.isDarwin(self.log):
          yield 'mpiCC'
        yield 'mpic++'
        yield 'mpCC'
        yield 'mpxlC'
      else:
        #attempt to match c++ compiler with c compiler
        if self.CC.find('win32fe_cl') >= 0:
          yield self.CC
        elif self.CC.find('win32fe_icl') >= 0:
          yield self.CC
        elif self.CC == 'gcc':
          yield 'g++'
        elif self.CC == 'clang':
          yield 'clang++'
        elif self.CC == 'icc':
          yield 'icpc'
        elif self.CC == 'xlc':
          yield 'xlC'
        elif self.CC == 'ncc':
          yield 'nc++'
        yield 'g++'
        yield 'clang++'
        yield 'c++'
        yield 'icpc'
        yield 'CC'
        yield 'cxx'
        yield 'cc++'
        yield 'xlC'
        yield 'ccpc'
        path = os.path.join(os.getcwd(),'lib','petsc','win32fe','bin')
        yield os.path.join(path,'win32fe_icl')
        yield os.path.join(path,'win32fe_cl')
        yield 'pgCC'
        yield 'CC'
    return

  def checkCxxCompiler(self):
    '''Locate a functional Cxx compiler'''
    self.mesg = ''
    for compiler in self.generateCxxCompilerGuesses():
      # Determine an acceptable extensions for the C++ compiler
      for ext in ['.cc', '.cpp', '.C']:
        self.framework.getCompilerObject('Cxx').sourceExtension = ext
        try:
          if self.getExecutable(compiler, resultName = 'CXX'):
            self.checkCompiler('Cxx')
            break
        except RuntimeError as e:
          self.mesg = str(e)
          self.logPrint('Error testing C++ compiler: '+str(e))
          self.showMPIWrapper(compiler)
          self.delMakeMacro('CXX')
          del self.CXX
      if hasattr(self, 'CXX'):
        try:
          self.executeShellCommand(self.CXX+' --version', log = self.log)
        except:
          pass
        break
    return

  def generateCxxPreprocessorGuesses(self):
    '''Determines the Cxx preprocessor from CXXPP, then --with-cxxpp, then the Cxx compiler'''
    if 'with-cxxpp' in self.argDB:
      yield self.argDB['with-cxxpp']
    elif 'CXXPP' in self.argDB:
      yield self.argDB['CXXPP']
    else:
      yield self.CXX+' -E'
      yield self.CXX+' --use cpp32'
    return

  def checkCxxPreprocessor(self):
    '''Locate a functional Cxx preprocessor'''
    if not hasattr(self,'CXX'): # pointless, it is checked already
      return
    with self.Language('Cxx'):
      for compiler in self.generateCxxPreprocessorGuesses():
        try:
          if self.getExecutable(compiler, resultName = 'CXXPP'):
            if not self.checkPreprocess('#include <cstdlib>\n'):
              raise RuntimeError('Cannot preprocess Cxx with '+self.CXXPP+'.')
            break
        except RuntimeError as e:
          self.logPrint(str(e))
          if os.path.basename(self.CXXPP) in ['mpicxx', 'mpiCC']:
            self.logPrint('MPI installation '+self.getCompiler()+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI')
          self.delMakeMacro('CXXPP')
          del self.CXXPP
    return


  def generateFortranCompilerGuesses(self):
    '''Determine the Fortran compiler'''

    if hasattr(self, 'FC'):
      yield self.FC
      if self.argDB['download-mpich']: mesg ='with downloaded MPICH'
      elif self.argDB['download-openmpi']: mesg ='with downloaded Open MPI'
      else: mesg = ''
      raise RuntimeError('Error '+mesg+': '+self.mesg)
    elif 'with-fc' in self.argDB:
      yield self.argDB['with-fc']
      raise RuntimeError('Fortran compiler you provided with --with-fc='+self.argDB['with-fc']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif 'FC' in self.argDB:
      yield self.argDB['FC']
      yield self.argDB['FC']
      raise RuntimeError('Fortran compiler you provided with -FC='+self.argDB['FC']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif self.usedMPICompilers and 'with-mpi-dir' in self.argDB and os.path.isdir(os.path.join(self.argDB['with-mpi-dir'], 'bin')):
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpinfort')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpiifort')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpif90')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpf90')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpxlf95_r')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpxlf90_r')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpxlf_r')
      if os.path.isfile(os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpif90')):
        raise RuntimeError('bin/mpif90 you provided with --with-mpi-dir='+self.argDB['with-mpi-dir']+' cannot be found or does not work.\nRun with --with-fc=0 if you wish to use this MPI and disable Fortran. See https://petsc.org/release/faq/#invalid-mpi-compilers')
    else:
      if self.usedMPICompilers:
        # TODO: Should only look for the MPI Fortran compiler related to the found MPI C compiler
        cray = os.getenv('CRAYPE_DIR')
        if cray:
          cross_fc = self.crayCrossCompiler('ftn')
          if cross_fc:
            self.cross_fc = cross_fc
            self.log.write('Cray system using Fortran cross compiler:'+cross_fc+'\n')
          yield 'ftn'
          if cross_fc: delattr(self, 'cross_fc')
        yield 'mpinfort'
        yield 'mpif90'
        yield 'mpiifort'
        yield 'mpxlf_r'
        yield 'mpxlf'
        yield 'mpf90'
      else:
        path = os.path.join(os.getcwd(),'lib','petsc','win32fe','bin')
        #attempt to match fortran compiler with c compiler
        if self.CC == 'gcc':
          yield 'gfortran'
        elif self.CC == 'clang':
          yield 'gfortran'
        elif self.CC == 'icc':
          yield 'ifort'
        elif self.CC == 'xlc':
          yield 'xlf90'
          yield 'xlf'
        elif self.CC == 'ncc':
          yield 'nfort'
        elif self.CC.find('win32fe_icl') >= 0:
          yield os.path.join(path,'win32fe_ifort')
        yield 'gfortran'
        yield 'g95'
        yield 'xlf90'
        yield 'xlf'
        yield 'f90'
        yield 'lf95'
        yield os.path.join(path,'win32fe_ifort')
        yield 'ifort'
        yield 'ifc'
        yield 'pgf90'
        yield 'f95'
        yield 'f90'
    return

  def checkFortranCompiler(self):
    '''Locate a functional Fortran compiler'''
    if 'with-fc' in self.argDB and self.argDB['with-fc'] == '0':
      if 'FC' in self.argDB:
        del self.argDB['FC']
      return
    self.mesg = ''
    for compiler in self.generateFortranCompilerGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'FC'):
          self.checkCompiler('FC')
          break
      except RuntimeError as e:
        self.mesg = str(e)
        self.logPrint('Error testing Fortran compiler: '+str(e))
        self.showMPIWrapper(compiler)
        self.delMakeMacro('FC')
        del self.FC
    if hasattr(self, 'FC'):
      try:
        self.executeShellCommand(self.FC+' --version', log = self.log)
      except:
        pass
    return

  def generateFortranPreprocessorGuesses(self):
    '''Determines the Fortran preprocessor from FPP, then --with-fpp, then the Fortran compiler'''
    if 'with-fpp' in self.argDB:
      yield self.argDB['with-fpp']
    elif 'FPP' in self.argDB:
      yield self.argDB['FPP']
    else:
      yield self.FC+' -E'
      yield self.FC+' --use cpp32'
    return

  def checkFortranPreprocessor(self):
    '''Locate a functional Fortran preprocessor'''
    if not hasattr(self, 'FC'):
      return
    with self.Language('FC'):
      for compiler in self.generateFortranPreprocessorGuesses():
        try:
          if self.getExecutable(compiler, resultName = 'FPP'):
            if not self.checkPreprocess('#define foo 10\n'):
              raise RuntimeError('Cannot preprocess Fortran with '+self.FPP+'.')
            break
        except RuntimeError as e:
          self.logPrint(str(e))
          if os.path.basename(self.FPP) in ['mpif90']:
            self.logPrint('MPI installation '+self.getCompiler()+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI')
          self.delMakeMacro('FPP')
          del self.FPP
    return

  def checkFortranComments(self):
    '''Make sure fortran comment "!" works'''
    self.pushLanguage('FC')
    if not self.checkCompile('! comment'):
      raise RuntimeError(self.getCompiler()+' cannot process fortran comments.')
    self.logPrint('Fortran comments can use ! in column 1')
    self.popLanguage()
    return


  def containsInvalidFlag(self, output):
    '''If the output contains evidence that an invalid flag was used, return True'''
    substrings = ('unknown argument', 'ignoring unsupported linker flag', 'unrecognized command line option','unrecognised command line option',
                  'unrecognized option','unrecognised option','not recognized',
                  'not recognised','unknown option','unknown warning option',
                  'unknown flag','unknown switch','ignoring option','ignored','argument unused',
                  'unsupported command line options encountered',
                  'not supported','is unsupported and will be skipped','illegal option',
                  'invalid option','invalid suboption','bad ',' option','petsc error',
                  'unbekannte option','linker input file unused because linking not done',
                  'warning: // comments are not allowed in this language',
                  'no se reconoce la opci','non reconnue','warning: unsupported linker arg:','ignoring unknown option')
    outlo = output.lower()
    return any(sub.lower() in outlo for sub in substrings)

  def containsInvalidLinkerFlag(self, output):
    '''If the output contains evidence that an invalid flag was used, return True'''
    substrings = ('unknown argument', 'ignoring unsupported linker flag', 'unrecognized command line option','unrecognised command line option',
                  'unrecognized option','unrecognised option','unknown option',
                  'unknown flag','unsupported command line options encountered',
                  'not supported','is unsupported and will be skipped','illegal option',
                  'invalid option','invalid suboption',
                  'unbekannte option',
                  'warning: -commons use_dylibs is no longer supported, using error treatment instead',
                  'warning: -bind_at_load is deprecated on macOS',
                  'no se reconoce la opci','non reconnue','warning: unsupported linker arg:','ignoring unknown option')
    outlo = output.lower()
    return any(sub.lower() in outlo for sub in substrings)

  def checkCompilerFlag(self, flag, includes = '', body = '', compilerOnly = 0):
    '''Determine whether the compiler accepts the given flag'''
    flagsArg = self.getCompilerFlagsArg(compilerOnly)
    oldFlags = getattr(self, flagsArg)
    setattr(self, flagsArg, oldFlags+' '+flag)
    (output, error, status) = self.outputCompile(includes, body)
    output = self.filterCompileOutput(output+'\n'+error,flag=flag)
    self.logPrint('Output from compiling with '+oldFlags+' '+flag+'\n'+output)
    setattr(self, flagsArg, oldFlags)
    # Please comment each entry and provide an example line
    if status:
      self.logPrint('Rejecting compiler flag '+flag+' due to nonzero status from link')
      return False
    elif self.containsInvalidFlag(output):
      self.logPrint('Rejecting compiler flag '+flag+' due to \n'+output)
      return False
    return True

  def insertCompilerFlag(self, flag, compilerOnly):
    '''DANGEROUS: Put in the compiler flag without checking'''
    if not flag: return
    flagsArg = self.getCompilerFlagsArg(compilerOnly)
    setattr(self, flagsArg, getattr(self, flagsArg)+' '+flag)
    self.log.write('Added '+self.language[-1]+' compiler flag '+flag+'\n')
    return

  def addCompilerFlag(self, flag, includes = '', body = '', extraflags = '', compilerOnly = 0):
    '''Determine whether the compiler accepts the given flag, and add it if valid, otherwise throw an exception'''
    if self.checkCompilerFlag(flag+' '+extraflags, includes, body, compilerOnly):
      self.insertCompilerFlag(flag, compilerOnly)
      return
    raise RuntimeError('Bad compiler flag: '+flag)

  @contextlib.contextmanager
  def extraCompilerFlags(self, extraFlags, lang = None, **kwargs):
    assert isinstance(extraFlags,(list,tuple)), "extraFlags must be either a list or tuple"
    if lang:
      self.pushLanguage(lang)
    flagsArg  = self.getCompilerFlagsArg()
    oldCompilerFlags = getattr(self,flagsArg)
    skipFlags = []
    try:
      for i,flag in enumerate(extraFlags):
        try:
          self.addCompilerFlag(flag, **kwargs)
        except RuntimeError:
          skipFlags.append((i,flag))
      yield skipFlags
    finally:
      # This last finally is a bit of deep magic, it makes it so that if the code in the
      # resulting yield throws some unrelated exception which is meant to be caught
      # outside this ctx manager then the flags and languages are still reset
      if lang:
        oldLang = self.popLanguage()
      setattr(self,flagsArg,oldCompilerFlags)

  def checkPragma(self):
    '''Check for all available applicable languages whether they complain (including warnings!) about potentially unknown pragmas'''
    usePragma = {}
    langMap = {'C':'CC','Cxx':'CXX','CUDA':'CUDAC','HIP':'HIPC','SYCL':'SYCLC'}
    for lang in langMap:
      if hasattr(self,langMap[lang]):
        usePragma[lang] = False
    for lang in usePragma.keys():
      with self.Language(lang):
        with self.extraCompilerFlags(['-Wunknown-pragmas']) as skipFlags:
          if not skipFlags:
            usePragma[lang] = self.checkCompile('#pragma GCC poison TEST')
    if all(usePragma.values()): self.framework.enablepoison = True
    return

  def generatePICGuesses(self):
    if self.language[-1] == 'CUDA':
      yield '-Xcompiler -fPIC'
      yield '-fPIC'
      return
    if config.setCompilers.Configure.isGNU(self.getCompiler(), self.log):
      PICFlags = ['-fPIC']
    elif config.setCompilers.Configure.isIBM(self.getCompiler(), self.log):
      PICFlags = ['-qPIC']
    else:
      PICFlags = ['-PIC','-qPIC','-KPIC','-fPIC','-fpic']
    try:
      output = self.executeShellCommand(self.getCompiler() + ' -show', log = self.log)[0]
    except:
      self.logPrint('Skipping checking MPI compiler command for PIC flag since MPI compiler -show causes an exception so is likely not an MPI compiler')
      output = ''
    output = output + ' ' + getattr(self, self.getCompilerFlagsArg(1)) + ' '
    # Try without specific PIC flag only if the MPI compiler or user compiler flag already provides a PIC option
    for i in PICFlags:
      if output.find(' '+i+' ') > -1:
        self.logPrint('Trying no specific compiler flag for PIC code since MPI compiler or current flags seem to provide such a flag with '+i)
        yield ''
        break
    for i in PICFlags:
      yield i
    yield ''

  def checkPIC(self):
    '''Determine the PIC option for each compiler'''
    self.usePIC = 0
    useSharedLibraries = 'with-shared-libraries' in self.argDB and self.argDB['with-shared-libraries']
    myLanguage = self.language[-1]
    if not self.argDB['with-pic'] and not useSharedLibraries:
      self.logPrint("Skip checking PIC options on user request")
      return
    if self.argDB['with-pic'] and not useSharedLibraries:
      # this is a flaw in configure; it is a legitimate use case where PETSc is built with PIC flags but not shared libraries
      # to fix it the capability to build shared libraries must be enabled in configure if --with-pic=true even if shared libraries are off and this
      # test must use that capability instead of using the default shared library build in that case which is static libraries
      raise RuntimeError("Cannot determine compiler PIC flags if shared libraries is turned off\nEither run using --with-shared-libraries or --with-pic=0 and supply the compiler PIC flag via CFLAGS, CXXFLAGS, and FCFLAGS\n")
    if self.sharedLibraries and self.mainLanguage == 'C': languages = []
    else: languages = ['C']
    langMap = {'FC':'FC','Cxx':'CXX','CUDA':'CUDAC','HIP':'HIPC','SYCL':'SYCLC'}
    for language in langMap:
      if hasattr(self,langMap[language]): languages.append(language)
    for language in languages:
      self.pushLanguage(language)
      if language in ['C','Cxx','CUDA','HIP','SYCL']:
        includeLine = _picTestIncludes()
      else:
        includeLine = '      function foo(a)\n      real:: a,x,bar\n      common /xx/ x\n      x=a\n      foo = bar(x)\n      end\n'
      compilerFlagsArg = self.getCompilerFlagsArg(1) # compiler only
      oldCompilerFlags = getattr(self, compilerFlagsArg)
      for testFlag in self.generatePICGuesses():
        if testFlag:
          self.logPrint('Trying '+language+' compiler flag '+testFlag+' for PIC code')
        else:
          self.logPrint('Trying '+language+' for PIC code without any compiler flag')
        acceptedPIC = 1
        try:
          self.addCompilerFlag(testFlag, compilerOnly = 1)
          acceptedPIC = self.checkLink(includes = includeLine, body = None, codeBegin = '', codeEnd = '', cleanup = 1, shared = 1, linkLanguage = myLanguage)
        except RuntimeError:
          acceptedPIC = 0
        if not acceptedPIC:
          self.logPrint('Rejected '+language+' compiler flag '+testFlag+' because shared linker cannot handle it')
          setattr(self, compilerFlagsArg, oldCompilerFlags)
          continue
        if testFlag:
          self.logPrint('Accepted '+language+' compiler flag '+testFlag+' for PIC code')
        else:
          self.logPrint('Accepted '+language+' PIC code without compiler flag')
        self.isPIC = 1
        break
      self.popLanguage()
    return

  def checkKandRFlags(self):
    '''Check C compiler flags that allow compiling K and R code (needed for some external packages)'''
    self.KandRFlags = []
    with self.Language('C'):
      if config.setCompilers.Configure.isGNU(self.getCompiler(), self.log) or config.setCompilers.Configure.isClang(self.getCompiler(), self.log):
        for f in ['-Wno-implicit-int', '-Wno-int-conversion', '-Wno-implicit-function-declaration', '-Wno-deprecated-non-prototype', '-fno-common']:
          if self.checkCompilerFlag(f, compilerOnly = 1):
            self.KandRFlags.append(f)

  def checkLargeFileIO(self):
    '''check for large file support with 64-bit offset'''
    if not self.argDB['with-large-file-io']:
      return
    languages = ['C']
    if hasattr(self, 'CXX'):
      languages.append('Cxx')
    for language in languages:
      self.pushLanguage(language)
      if self.checkCompile('#include <unistd.h>','#ifndef _LFS64_LARGEFILE \n#error no largefile defines \n#endif'):
        try:
          self.addCompilerFlag('-D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64',compilerOnly=1)
        except RuntimeError as e:
          self.logPrint('Error adding ' +language+ ' flags -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64')
      else:
        self.logPrint('Rejected ' +language+ ' flags -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64')
      self.popLanguage()
    return

  def getArchiverFlags(self, archiver):
    prog = os.path.basename(archiver).split(' ')[0]
    flag = ''
    if 'AR_FLAGS' in self.argDB:
      flag = self.argDB['AR_FLAGS']
    elif prog.endswith('ar'):
      flag = 'cr'
    elif os.path.basename(archiver).endswith('_lib'):
      flag = '-a'
    if prog.endswith('ar') and not (self.isSolarisAR(prog, self.log) or self.isAIXAR(prog, self.log)):
      self.FAST_AR_FLAGS = 'Scq'
    else:
      self.FAST_AR_FLAGS = flag
    self.framework.addMakeMacro('FAST_AR_FLAGS',self.FAST_AR_FLAGS )
    return flag

  def generateArchiverGuesses(self):
    defaultAr = None
    if 'with-ar' in self.argDB:
      defaultAr = self.argDB['with-ar']
    envAr = None
    if 'AR' in self.argDB:
      envAr = self.argDB['AR']
    defaultRanlib = None
    if 'with-ranlib' in self.argDB:
      defaultRanlib = self.argDB['with-ranlib']
    envRanlib = None
    if 'RANLIB' in self.argDB:
      envRanlib = self.argDB['RANLIB']
    if defaultAr and defaultRanlib:
      yield(defaultAr,self.getArchiverFlags(defaultAr),defaultRanlib)
      raise RuntimeError('The archiver set --with-ar="'+defaultAr+'" is broken or incompatible with the ranlib set --with-ranlib="'+defaultRanlib+'".')
    if defaultAr and envRanlib:
      yield(defaultAr,self.getArchiverFlags(defaultAr),envRanlib)
      raise RuntimeError('The archiver set --with-ar="'+defaultAr+'" is broken or incompatible with the ranlib set (perhaps in your environment) -RANLIB="'+envRanlib+'".')
    if envAr and defaultRanlib:
      yield(envAr,self.getArchiverFlags(envAr),defaultRanlib)
      raise RuntimeError('The archiver set --AR="'+envAr+'" is broken or incompatible with the ranlib set --with-ranlib="'+defaultRanlib+'".')
    if envAr and envRanlib:
      yield(envAr,self.getArchiverFlags(envAr),envRanlib)
      raise RuntimeError('The archiver set --AR="'+envAr+'" is broken or incompatible with the ranlib set (perhaps in your environment) -RANLIB="'+envRanlib+'".')
    if defaultAr:
      yield (defaultAr,self.getArchiverFlags(defaultAr),'ranlib')
      yield (defaultAr,self.getArchiverFlags(defaultAr),'true')
      raise RuntimeError('You set a value for --with-ar='+defaultAr+'", but '+defaultAr+' cannot be used\n')
    if envAr:
      yield (envAr,self.getArchiverFlags(envAr),'ranlib')
      yield (envAr,self.getArchiverFlags(envAr),'true')
      raise RuntimeError('You set a value for -AR="'+envAr+'" (perhaps in your environment), but '+envAr+' cannot be used\n')
    if defaultRanlib:
      yield ('ar',self.getArchiverFlags('ar'),defaultRanlib)
      path = os.path.join(os.getcwd(),'lib','petsc','bin')
      war  = os.path.join(path,'win32fe_lib')
      yield (war,self.getArchiverFlags(war),defaultRanlib)
      raise RuntimeError('You set --with-ranlib="'+defaultRanlib+'", but '+defaultRanlib+' cannot be used\n')
    if envRanlib:
      yield ('ar',self.getArchiverFlags('ar'),envRanlib)
      path = os.path.join(os.getcwd(),'lib','petsc','bin')
      war  = os.path.join(path,'win32fe_lib')
      yield (war,self.getArchiverFlags('war'),envRanlib)
      raise RuntimeError('You set -RANLIB="'+envRanlib+'" (perhaps in your environment), but '+defaultRanlib+' cannot be used\n')
    if config.setCompilers.Configure.isWindows(self.getCompiler(), self.log):
      path = os.path.join(os.getcwd(),'lib','petsc','bin')
      war  = os.path.join(path,'win32fe_lib')
      yield (war,self.getArchiverFlags(war),'true')
    yield ('ar',self.getArchiverFlags('ar'),'ranlib -c')
    yield ('ar',self.getArchiverFlags('ar'),'ranlib')
    yield ('ar',self.getArchiverFlags('ar'),'true')
    # IBM with 64-bit pointers
    yield ('ar','-X64 '+self.getArchiverFlags('ar'),'ranlib -c')
    yield ('ar','-X64 '+self.getArchiverFlags('ar'),'ranlib')
    yield ('ar','-X64 '+self.getArchiverFlags('ar'),'true')
    return

  def checkArchiver(self):
    '''Check that the archiver exists and can make a library usable by the compiler'''
    objName    = os.path.join(self.tmpDir, 'conf1.o')
    arcUnix    = os.path.join(self.tmpDir, 'libconf1.a')
    arcWindows = os.path.join(self.tmpDir, 'libconf1.lib')
    def checkArchive(command, status, output, error):
      if error:
        error = error.splitlines()
        error = [s for s in error if not (s.find('unsupported GNU_PROPERTY_TYPE') >= 0 and s.find('warning:') >= 0)]
        error = [s for s in error if s.find("xiar: executing 'ar'") < 0]
        if error: error = '\n'.join(error)
        else: error = ''
      if error or status:
        self.logError('archiver', status, output, error)
        if os.path.isfile(objName):
          os.remove(objName)
        raise RuntimeError('Archiver is not functional')
      return
    def checkRanlib(command, status, output, error):
      if error or status:
        self.logError('ranlib', status, output, error)
        if os.path.isfile(arcUnix):
          os.remove(arcUnix)
        raise RuntimeError('Ranlib is not functional with your archiver.  Try --with-ranlib=true if ranlib is unnecessary.')
      return
    oldLibs = self.LIBS
    self.pushLanguage('C')
    for (archiver, arflags, ranlib) in self.generateArchiverGuesses():
      if not self.checkCompile('', 'int foo(int a) {\n  return a+1;\n}\n\n', cleanup = 0, codeBegin = '', codeEnd = ''):
        raise RuntimeError('Compiler is not functional')
      if os.path.isfile(objName):
        os.remove(objName)
      os.rename(self.compilerObj, objName)
      if self.getExecutable(archiver, getFullPath = 1, resultName = 'AR'):
        if self.getExecutable(ranlib, getFullPath = 1, resultName = 'RANLIB'):
          arext = 'a'
          try:
            (output, error, status) = config.base.Configure.executeShellCommand(self.AR+' '+arflags+' '+arcUnix+' '+objName, checkCommand = checkArchive, log = self.log)
            (output, error, status) = config.base.Configure.executeShellCommand(self.RANLIB+' '+arcUnix, checkCommand = checkRanlib, log = self.log)
          except RuntimeError as e:
            self.logPrint(str(e))
            continue
          self.LIBS = '-L'+self.tmpDir+' -lconf1 ' + oldLibs
          success =  self.checkLink('extern int foo(int);', '  int b = foo(1);  (void)b')
          os.rename(arcUnix, arcWindows)
          if not success:
            arext = 'lib'
            success = self.checkLink('extern int foo(int);', '  int b = foo(1);  (void)b')
            os.remove(arcWindows)
            if success:
              break
          else:
            os.remove(arcWindows)
            break
      else:
        if os.path.isfile(objName):
          os.remove(objName)
        self.LIBS = oldLibs
        self.popLanguage()
        if 'with-ar' in self.argDB:
          raise RuntimeError('Archiver set with --with-ar='+self.argDB['with-ar']+' does not exist')
        else:
          raise RuntimeError('Could not find a suitable archiver.  Use --with-ar to specify an archiver.')
    self.AR_FLAGS      = arflags
    self.AR_LIB_SUFFIX = arext
    self.framework.addMakeMacro('AR_FLAGS', self.AR_FLAGS)
    self.addMakeMacro('AR_LIB_SUFFIX', self.AR_LIB_SUFFIX)
    os.remove(objName)
    self.LIBS = oldLibs
    self.popLanguage()
    return

  def checkArchiverRecipeArgfile(self):
    '''Checks if AR handles @ notation'''
    def checkArchiverArgfile(command, status, output, error):
      if error or status:
        self.logError('archiver', status, output, error)
        if os.path.isfile(objName):
          os.remove(objName)
        raise RuntimeError('ArchiverArgfile error')
      return
    oldDir = os.getcwd()
    os.chdir(self.tmpDir)
    try:
      objName = 'checkRecipeArgfile.o'
      obj = open(objName, 'a').close()
      argsName = 'checkRecipeArgfile.args'
      args = open(argsName, 'a')
      args.write(objName)
      args.close()
      archiveName = 'checkRecipeArgfile.'+self.AR_LIB_SUFFIX
      (output, error, status) = config.base.Configure.executeShellCommand(self.AR+' '+self.AR_FLAGS+' '+archiveName+' @'+argsName,checkCommand = checkArchiverArgfile, log = self.log)
      os.remove(objName)
      os.remove(argsName)
      os.remove(archiveName)
      if not status:
        self.framework.addMakeMacro('AR_ARGFILE','yes')
    except RuntimeError:
      pass
    os.chdir(oldDir)

  def setStaticLinker(self):
    language = self.language[-1]
    return self.framework.setSharedLinkerObject(language, self.framework.getLanguageModule(language).StaticLinker(self.argDB))

  def generateSharedLinkerGuesses(self):
    if not self.argDB['with-shared-libraries']:
      self.setStaticLinker()
      self.staticLinker = self.AR
      self.staticLibraries = 1
      self.LDFLAGS = ''
      yield (self.AR, [], self.AR_LIB_SUFFIX)
      raise RuntimeError('Archiver failed static link check')
    if 'with-shared-ld' in self.argDB:
      yield (self.argDB['with-shared-ld'], [], 'so')
    if 'LD_SHARED' in self.argDB:
      yield (self.argDB['LD_SHARED'], [], 'so')
    if Configure.isDarwin(self.log):
      if 'with-shared-ld' in self.argDB:
        yield (self.argDB['with-shared-ld'], ['-dynamiclib', '-undefined dynamic_lookup', '-no_compact_unwind'], 'dylib')
      if hasattr(self, 'CXX') and self.mainLanguage == 'Cxx':
        yield (self.CXX, ['-dynamiclib', '-undefined dynamic_lookup', '-no_compact_unwind'], 'dylib')
      yield (self.CC, ['-dynamiclib', '-undefined dynamic_lookup', '-no_compact_unwind'], 'dylib')
    if hasattr(self, 'CXX') and self.mainLanguage == 'Cxx':
      # C++ compiler default
      yield (self.CXX, ['-qmkshrobj'], 'so')
      yield (self.CXX, ['-shared'], 'so')
      yield (self.CXX, ['-dynamic'], 'so')
      yield (self.CC, ['-shared'], 'dll')
    # C compiler default
    yield (self.CC, ['-qmkshrobj'], 'so')
    yield (self.CC, ['-shared'], 'so')
    yield (self.CC, ['-dynamic'], 'so')
    yield (self.CC, ['-shared'], 'dll')
    # Windows default
    if self.CC.find('win32fe') >=0:
      if hasattr(self, 'CXX') and self.mainLanguage == 'Cxx':
        yield (self.CXX, ['-LD'], 'dll')
      yield (self.CC, ['-LD'], 'dll')
    # Solaris default
    if Configure.isSolaris(self.log):
      if hasattr(self, 'CXX') and self.mainLanguage == 'Cxx':
        yield (self.CXX, ['-G'], 'so')
      yield (self.CC, ['-G'], 'so')
    # If user does not explicitly enable shared-libraries - disable shared libraries and default to static linker
    if not 'with-shared-libraries' in self.framework.clArgDB:
      self.argDB['with-shared-libraries'] = 0
      self.setStaticLinker()
      self.staticLinker = self.AR
      self.staticLibraries = 1
      self.LDFLAGS = ''
      yield (self.AR, [], self.AR_LIB_SUFFIX)
    raise RuntimeError('Exhausted all shared linker guesses. Could not determine how to create a shared library!')

  def checkSharedLinker(self):
    '''Check that the linker can produce shared libraries'''
    self.sharedLibraries = 0
    self.staticLibraries = 0
    for linker, flags, ext in self.generateSharedLinkerGuesses():
      self.logPrint('Checking shared linker '+linker+' using flags '+str(flags))
      if self.getExecutable(linker, resultName = 'LD_SHARED'):
        for picFlag in self.generatePICGuesses():
          self.logPrint('Trying '+self.language[-1]+' compiler flag '+picFlag)
          compilerFlagsArg = self.getCompilerFlagsArg(1) # compiler only
          oldCompilerFlags = getattr(self, compilerFlagsArg)
          accepted = 1
          try:
            self.addCompilerFlag(picFlag,compilerOnly=1)
          except RuntimeError:
            accepted = 0
          if accepted:
            goodFlags = list(filter(self.checkLinkerFlag, flags))
            self.sharedLinker = self.LD_SHARED
            self.sharedLibraryFlags = goodFlags
            self.sharedLibraryExt = ext
            if ext == 'dll':
              dllexport = '__declspec(dllexport) '
              dllimport = '__declspec(dllimport) '
            else:
              dllexport = ''
              dllimport = ''
            # using printf appears to correctly identify non-pic code on X86_64
            if self.checkLink(includes = _picTestIncludes(dllexport), codeBegin = '', codeEnd = '', cleanup = 0, shared = 1):
              oldLib  = self.linkerObj
              oldLibs = self.LIBS
              self.LIBS += ' -L'+self.tmpDir+' -lconftest'
              accepted = self.checkLink(includes = dllimport+'int foo(void);', body = 'int ret = foo();\nif (ret) {}\n')
              os.remove(oldLib)
              self.LIBS = oldLibs
              if accepted:
                self.sharedLibraries = 1
                self.logPrint('Using shared linker '+self.sharedLinker+' with flags '+str(self.sharedLibraryFlags)+' and library extension '+self.sharedLibraryExt)
                break
          self.logPrint('Rejected '+self.language[-1]+' compiler flag '+picFlag+' because it was not compatible with shared linker '+linker+' using flags '+str(flags))
          setattr(self, compilerFlagsArg, oldCompilerFlags)
        if os.path.isfile(self.linkerObj): os.remove(self.linkerObj)
        if self.sharedLibraries: break
        self.delMakeMacro('LD_SHARED')
        del self.LD_SHARED
        if hasattr(self,'sharedLinker'): del self.sharedLinker
    return

  def checkLinkerFlag(self, flag):
    '''Determine whether the linker accepts the given flag'''
    flagsArg = self.getLinkerFlagsArg()
    oldFlags = getattr(self, flagsArg)
    setattr(self, flagsArg, oldFlags+' '+flag)
    (output, status) = self.outputLink('', '')
    valid = 1
    if status:
      valid = 0
      self.logPrint('Rejecting linker flag '+flag+' due to nonzero status from link')
    if self.containsInvalidLinkerFlag(output):
      valid = 0
      self.logPrint('Rejecting '+self.language[-1]+' linker flag '+flag+' due to \n'+output)
    if valid:
      self.logPrint('Valid '+self.language[-1]+' linker flag '+flag)
    setattr(self, flagsArg, oldFlags)
    return valid

  def addLinkerFlag(self, flag):
    '''Determine whether the linker accepts the given flag, and add it if valid, otherwise throw an exception'''
    if self.checkLinkerFlag(flag):
      flagsArg = self.getLinkerFlagsArg()
      setattr(self, flagsArg, getattr(self, flagsArg)+' '+flag)
      return
    raise RuntimeError('Bad linker flag: '+flag)

  def checkLinkerMac(self):
    '''Tests some Apple Mac specific linker flags'''
    self.addDefine('USING_DARWIN', 1)
    langMap = {'C':'CC','FC':'FC','Cxx':'CXX','CUDA':'CUDAC','HIP':'HIPC','SYCL':'SYCLC'}
    languages = ['C']
    if hasattr(self, 'CXX'):
      languages.append('Cxx')
    if hasattr(self, 'FC'):
      languages.append('FC')
    ldTestFlags = ['-Wl,-bind_at_load', '-Wl,-commons,use_dylibs', '-Wl,-search_paths_first', '-Wl,-no_compact_unwind']
    if self.LDFLAGS.find('-Wl,-ld_classic') < 0:
      ldTestFlags.append('-Wl,-no_warn_duplicate_libraries')
    for language in languages:
      self.pushLanguage(language)
      for testFlag in ldTestFlags:
        if self.checkLinkerFlag(testFlag):
          # expand to CC_LINKER_FLAGS or CXX_LINKER_FLAGS or FC_LINKER_FLAGS
          linker_flag_var = langMap[language]+'_LINKER_FLAGS'
          val = getattr(self,linker_flag_var)
          val.append(testFlag)
          setattr(self,linker_flag_var,val)
          self.logPrint('Accepted macOS linker flag ' + testFlag)
        else:
          self.logPrint('Rejected macOS linker flag ' + testFlag)
      self.popLanguage()
    return

  def checkLinkerWindows(self):
    '''Turns off linker warning about unknown .o files extension'''
    langMap = {'C':'CC','FC':'FC','Cxx':'CXX','CUDA':'CUDAC','HIP':'HIPC','SYCL':'SYCLC'}
    languages = ['C']
    if hasattr(self, 'CXX'):
      languages.append('Cxx')
    for language in languages:
      self.pushLanguage(language)
      for testFlag in ['-Qwd10161']:  #Warning for Intel icl,  there appear to be no way to remove warnings with Microsoft cl
        if self.checkLinkerFlag(testFlag):
          # expand to CC_LINKER_FLAGS or CXX_LINKER_FLAGS or FC_LINKER_FLAGS
          linker_flag_var = langMap[language]+'_LINKER_FLAGS'
          val = getattr(self,linker_flag_var)
          val.append(testFlag)
          setattr(self,linker_flag_var,val)
      self.popLanguage()
    return

  def checkSharedLinkerPaths(self):
    '''Determine the shared linker path options
       - IRIX: -rpath
       - Linux, OSF: -Wl,-rpath,
       - Solaris: -R
       - FreeBSD: -Wl,-R,'''
    languages = ['C']
    if hasattr(self, 'CXX'):
      languages.append('Cxx')
    if hasattr(self, 'FC'):
      languages.append('FC')
    if hasattr(self, 'CUDAC'):
      languages.append('CUDA')
    if hasattr(self, 'HIPC'):
      languages.append('HIP')
    if hasattr(self, 'SYCLC'):
      languages.append('SYCL')
    for language in languages:
      flag = '-L'
      self.pushLanguage(language)
      if Configure.isCygwin(self.log):
        self.logPrint('Cygwin detected! disabling -rpath test.')
        testFlags = []
      # test '-R' before '-rpath' as sun compilers [c,fortran] don't give proper errors with wrong options.
      elif not Configure.isDarwin(self.log):
        testFlags = ['-Wl,-rpath,', '-R','-rpath ' , '-Wl,-R,']
      else:
        testFlags = ['-Wl,-rpath,']
      # test '-R' before '-Wl,-rpath' for SUN compilers [as cc on linux accepts -Wl,-rpath, but  f90 & CC do not.
      if self.isSun(self.framework.getCompiler(), self.log):
        testFlags.insert(0,'-R')
      for testFlag in testFlags:
        self.logPrint('Trying '+language+' linker flag '+testFlag)
        if self.checkLinkerFlag(testFlag+os.path.abspath(os.getcwd())):
          flag = testFlag
          break
        else:
          self.logPrint('Rejected '+language+' linker flag '+testFlag)
      self.popLanguage()
      setattr(self, language+'SharedLinkerFlag', flag)
    return

  def checkLibC(self):
    '''Test whether we need to explicitly include libc in shared linking
       - Mac OSX requires an explicit reference to libc for shared linking'''
    self.explicitLibc = None
    if self.staticLibraries:
      return
    tmpCompilerDefines   = self.compilerDefines
    self.compilerDefines = ''
    code = '#include <stdlib.h> \nint foo(void) {void *chunk = malloc(31); free(chunk); return 0;}\n'
    if self.checkLink(includes = code, codeBegin = '', codeEnd = '', shared = 1):
      self.logPrint('Shared linking does not require an explicit libc reference')
      self.compilerDefines = tmpCompilerDefines
      return
    oldLibs = self.LIBS
    self.LIBS += '-lc '
    if self.checkLink(includes = code, codeBegin = '', codeEnd = '', shared = 1):
      self.logPrint('Shared linking requires an explicit libc reference')
      self.compilerDefines = tmpCompilerDefines
      self.explicitLibc = ['libc.so']
      return
    self.LIBS = oldLibs
    self.compilerDefines = tmpCompilerDefines
    self.logPrintWarning('Shared linking may not function on this architecture')
    self.staticLibrary=1
    self.sharedLibrary=0

  def generateDynamicLinkerGuesses(self):
    if 'with-dynamic-ld' in self.argDB:
      yield (self.argDB['with-dynamic-ld'], [], 'so')
    # Mac OSX
    if Configure.isDarwin(self.log):
      if 'with-dynamic-ld' in self.argDB:
        yield (self.argDB['with-dynamic-ld'], ['-dynamiclib -undefined dynamic_lookup'], 'dylib')
      if hasattr(self, 'CXX') and self.mainLanguage == 'Cxx':
        yield (self.CXX, ['-dynamiclib -undefined dynamic_lookup'], 'dylib')
      yield (self.CC, ['-dynamiclib -undefined dynamic_lookup'], 'dylib')
    # Shared default
    if hasattr(self, 'sharedLinker'):
      yield (self.sharedLinker, self.sharedLibraryFlags, 'so')
    # C++ Compiler default
    if hasattr(self, 'CXX') and self.mainLanguage == 'Cxx':
      yield (self.CXX, ['-shared'], 'so')
    # C Compiler default
    yield (self.CC, ['-shared'], 'so')
    self.logPrint('Unable to find working dynamic linker')

  def checkDynamicLinker(self):
    '''Check that the linker can dynamically load shared libraries'''
    self.dynamicLibraries = 0
    if not self.headers.check('dlfcn.h'):
      self.logPrint('Dynamic loading disabled since dlfcn.h was missing')
      return
    self.libraries.saveLog()
    if not self.libraries.check('', ['dlopen', 'dlsym', 'dlclose']):
      if not self.libraries.add('dl', ['dlopen', 'dlsym', 'dlclose']):
        self.logWrite(self.libraries.restoreLog())
        self.logPrint('Dynamic linking disabled since functions dlopen(), dlsym(), and dlclose() were not found')
        return
    self.logWrite(self.libraries.restoreLog())
    for linker, flags, ext in self.generateDynamicLinkerGuesses():
      self.logPrint('Checking dynamic linker '+linker+' using flags '+str(flags))
      if self.getExecutable(linker, resultName = 'dynamicLinker'):
        flagsArg = self.getLinkerFlagsArg()
        goodFlags = list(filter(self.checkLinkerFlag, flags))
        self.dynamicLibraryFlags = goodFlags
        self.dynamicLibraryExt = ext
        testMethod = 'foo'
        if self.checkLink(includes = '#include <stdio.h>\nint '+testMethod+'(void) {printf("test");return 0;}\n', codeBegin = '', codeEnd = '', cleanup = 0, shared = 'dynamic'):
          oldLib  = self.linkerObj
          code = '''
void *handle = dlopen("%s", 0);
int (*foo)(void) = (int (*)(void)) dlsym(handle, "foo");

if (!foo) {
  printf("Could not load symbol\\n");
  return -1;
}
if ((*foo)()) {
  printf("Invalid return from foo()\\n");
  return -1;
}
if (dlclose(handle)) {
  printf("Could not close library\\n");
  return -1;
}
''' % oldLib
          if self.checkLink(includes = '#include <dlfcn.h>\n#include <stdio.h>', body = code):
            self.dynamicLibraries = 1
            self.logPrint('Using dynamic linker '+self.dynamicLinker+' with flags '+str(self.dynamicLibraryFlags)+' and library extension '+self.dynamicLibraryExt)
            os.remove(oldLib)
            break
        if os.path.isfile(self.linkerObj): os.remove(self.linkerObj)
        del self.dynamicLinker
    return

  def output(self):
    '''Output module data as defines and substitutions'''
    if hasattr(self, 'CC'):
      self.addSubstitution('CC', self.CC)
      self.addSubstitution('CFLAGS', self.CFLAGS)
      self.addMakeMacro('CC_LINKER_SLFLAG', self.CSharedLinkerFlag)
    if hasattr(self, 'CPP'):
      self.addSubstitution('CPP', self.CPP)
      self.addSubstitution('CPPFLAGS', self.CPPFLAGS)
    if hasattr(self, 'CUDAC'):
      self.addSubstitution('CUDAC', self.CUDAC)
      self.addSubstitution('CUDAFLAGS', self.CUDAFLAGS)
    if hasattr(self, 'CUDAPP'):
      self.addSubstitution('CUDAPP', self.CUDAPP)
      self.addSubstitution('CUDAPPFLAGS', self.CUDAPPFLAGS)
    if hasattr(self, 'HIPC'):
      self.addSubstitution('HIPC', self.HIPC)
      self.addSubstitution('HIPFLAGS', self.HIPFLAGS)
    if hasattr(self, 'HIPPP'):
      self.addSubstitution('HIPPP', self.HIPPP)
      self.addSubstitution('HIPPPFLAGS', self.HIPPPFLAGS)
    if hasattr(self, 'SYCLC'):
      self.addSubstitution('SYCLC', self.SYCLC)
      self.addSubstitution('SYCLFLAGS', self.SYCLFLAGS)
    if hasattr(self, 'SYCLPP'):
      self.addSubstitution('SYCLPP', self.SYCLPP)
      self.addSubstitution('SYCLPPFLAGS', self.SYCLPPFLAGS)
    if hasattr(self, 'CXX'):
      self.addSubstitution('CXX', self.CXX)
      self.addSubstitution('CXX_CXXFLAGS', self.CXX_CXXFLAGS)
      self.addSubstitution('CXXFLAGS', self.CXXFLAGS)
      self.addSubstitution('CXX_LINKER_SLFLAG', self.CxxSharedLinkerFlag)
    else:
      self.addSubstitution('CXX', '')
    if hasattr(self, 'CXXPP'):
      self.addSubstitution('CXXPP', self.CXXPP)
      self.addSubstitution('CXXPPFLAGS', self.CXXPPFLAGS)
    if hasattr(self, 'FC'):
      self.addSubstitution('FC', self.FC)
      self.addSubstitution('FFLAGS', self.FFLAGS)
      self.addMakeMacro('FC_LINKER_SLFLAG', self.FCSharedLinkerFlag)
    else:
      self.addSubstitution('FC', '')
    self.addSubstitution('LDFLAGS', self.LDFLAGS)
    if hasattr(self, 'FPP'):
      self.addSubstitution('FPP', self.FPP)
      self.addSubstitution('FPPFLAGS', self.FPPFLAGS)
    self.addSubstitution('LIBS', self.LIBS)
    if hasattr(self, 'sharedLibraryFlags'):
      self.addSubstitution('SHARED_LIBRARY_FLAG', ' '.join(self.sharedLibraryFlags))
    else:
      self.addSubstitution('SHARED_LIBRARY_FLAG','')
    return

  def updateMPICompilers(self, mpicc, mpicxx, mpifc):
    '''Reset compilers by an external module aka MPI'''
    self.CC = mpicc
    self.delMakeMacro("CC")

    if hasattr(self, 'CXX'):
      self.CXX = mpicxx
      self.delMakeMacro("CXX")

    if hasattr(self, 'FC'):
      self.FC = mpifc
      self.delMakeMacro("FC")

    self.configure()
    self.usedMPICompilers=1
    return

  def checkMPICompilerOverride(self):
    '''Check if --with-mpi-dir is used along with CC CXX or FC compiler options.
    This usually prevents mpi compilers from being used - so issue a warning'''

    if 'with-mpi-dir' in self.argDB and self.argDB['with-mpi-compilers']:
      optcplrs = [(['with-cc','CC'],['mpincc','mpiicc','mpicc','mpcc','hcc','mpcc_r']),
              (['with-fc','FC'],['mpinfort','mpiifort','mpif90','mpxlf95_r','mpxlf90_r','mpxlf_r','mpf90']),
              (['with-cxx','CXX'],['mpinc++','mpiicpc','mpicxx','hcp','mpic++','mpiCC','mpCC_r'])]
      for opts,cplrs in optcplrs:
        for opt in opts:
          if (opt in self.argDB  and self.argDB[opt] != '0'):
            # check if corresponding mpi wrapper exists
            for cplr in cplrs:
              for mpicplr in [os.path.join(self.argDB['with-mpi-dir'], 'bin', cplr),os.path.join(self.argDB['with-mpi-dir'], 'intel64', 'bin', cplr)]:
                if os.path.exists(mpicplr):
                  msg = '--'+opt+'='+self.argDB[opt]+' is specified along with --with-mpi-dir='+self.argDB['with-mpi-dir']+' which implies using '+mpicplr+'.\n\
  configure is confused and does not know which compiler to select and use! Please specify either [mpi] compilers or --with-mpi-dir - but not both!\n\
  In most cases, specifying --with-mpi-dir - and not explicitly listing compilers could be preferable.'
                  raise RuntimeError(msg)
    return

  def requireMpiLdPath(self):
    '''Open MPI wrappers require LD_LIBRARY_PATH set'''
    if 'with-mpi-dir' in self.argDB:
      libdir = os.path.join(self.argDB['with-mpi-dir'], 'lib')
      if os.path.exists(os.path.join(libdir,'libopen-rte.so')):
        Configure.addLdPath(libdir)
        self.logPrint('Adding to LD_LIBRARY_PATH '+libdir)
    return

  def resetEnvCompilers(self):
    '''Remove compilers from the shell environment so they do not interfere with testing'''
    ignoreEnvCompilers = ['CC','CXX','FC','F77','F90']
    ignoreEnv = ['CFLAGS','CXXFLAGS','FCFLAGS','FFLAGS','F90FLAGS','CPP','CPPFLAGS','CXXPP','CXXPPFLAGS','LDFLAGS','LIBS','MPI_DIR','RM','MAKEFLAGS','AR','RANLIB']
    for envVal in ignoreEnvCompilers + ignoreEnv:
      if envVal in os.environ:
        msg = 'Found environment variable: %s=%s. ' % (envVal, os.environ[envVal])
        if envVal in self.framework.clArgDB or (envVal in ignoreEnvCompilers and 'with-'+envVal.lower() in self.framework.clArgDB):
          self.logPrintWarning(msg+"Ignoring it, since it's also set on command line")
          del os.environ[envVal]
        elif self.argDB['with-environment-variables']:
          self.logPrintWarning(msg+'Using it! Use "./configure --disable-environment-variables" to NOT use the environmental variables')
        else:
          self.logPrintWarning(msg+'Ignoring it! Use "./configure %s=$%s" if you really want to use this value' % (envVal,envVal))
          del os.environ[envVal]
    return

  def checkEnvCompilers(self):
    '''Set configure compilers from the environment, from -with-environment-variables'''
    if 'with-environment-variables' in self.framework.clArgDB:
      envVarChecklist = ['CC','CFLAGS','CXX','CXXFLAGS','FC','FCFLAGS','F77','FFLAGS','F90','F90FLAGS','CPP','CPPFLAGS','CXXPP','CXXPPFLAGS','LDFLAGS','LIBS','MPI_DIR','RM','MAKEFLAGS','AR']
      for ev in envVarChecklist:
        if ev in os.environ:
          self.argDB[ev] = os.environ[ev]

    # abort if FCFLAGS and FFLAGS both set, but to different values
    if 'FFLAGS' in self.argDB and 'FCFLAGS' in self.argDB:
      if self.argDB['FCFLAGS'] != self.argDB['FFLAGS']:
        raise RuntimeError('FCFLAGS and FFLAGS are both set, but with different values (FCFLAGS=%s, FFLAGS=%s)'%(self.argDB['FCFLAGS'],self.argDB['FFLAGS']))
    return

  def checkIntoShared(self,symbol,lib):
    '''Check that a given library can be linked into a shared library'''
    import sys
    if not self.checkCompile(includes = 'char *'+symbol+'(void);\n',body = 'return '+symbol+'();\n', cleanup = 0, codeBegin = 'char* testroutine(void){', codeEnd = '}'):
      raise RuntimeError('Unable to compile test file with symbol: '+symbol)
    oldLibs = self.LIBS
    self.LIBS = self.libraries.toStringNoDupes(lib) + ' '+self.LIBS
    ret = self.checkLink(includes = 'char *'+symbol+'(void);\n',body = 'return '+symbol+'();\n', cleanup = 0, codeBegin = 'char* testroutine(void){', codeEnd = '}',shared =1)
    self.LIBS = oldLibs
    return ret

  def checkAtFileOption(self):
    '''Check if linker supports @file option'''
    optfile = os.path.join(self.tmpDir,'optfile')
    with open(optfile,'w') as fd:
      fd.write(str(self.getCompilerFlags()))
    if self.checkLinkerFlag('@'+optfile):
      self.framework.addMakeMacro('PCC_AT_FILE',1)
    else:
      self.logPrint('@file option test failed!')
    return

  def configure(self):
    self.mainLanguage = self.languages.clanguage
    self.executeTest(self.resetEnvCompilers)
    self.executeTest(self.checkEnvCompilers)
    self.executeTest(self.checkMPICompilerOverride)
    self.executeTest(self.requireMpiLdPath)
    self.executeTest(self.checkInitialFlags)
    if hasattr(self.framework,'conda_active'):
      self.framework.additional_error_message = 'Conda may be causing this compiling/linking problem, consider turning off Conda.'
    self.executeTest(self.checkCCompiler)
    self.executeTest(self.checkCPreprocessor)

    for LANG in ['Cxx','CUDA','HIP','SYCL']:
      compilerName = LANG.upper() if LANG == 'Cxx' else LANG+'C'
      argdbName    = 'with-' + compilerName.casefold()
      argdbVal     = self.argDB.get(argdbName)
      if argdbVal == '0':
        # compiler was explicitly disabled, i.e. --with-cxx=0
        COMPILER_NAME = compilerName.upper()
        if COMPILER_NAME in self.argDB:
          del self.argDB[COMPILER_NAME]
          continue
      else:
        self.executeTest(getattr(self,LANG.join(('check','Compiler'))))
        try:
          self.executeTest(self.checkDeviceHostCompiler,args=[LANG])
        except NotImplementedError:
          pass
        if hasattr(self,compilerName):
          compiler = self.getCompiler(lang=LANG)
          isGNUish = self.isGNU(compiler,self.log) or self.isClang(compiler,self.log)
          try:
            self.executeTest(self.checkCxxDialect,args=[LANG],kargs={'isGNUish':isGNUish})
          except RuntimeError as e:
            self.mesg = str(e)
            if argdbVal is not None:
              # user explicitly enabled a compiler, e.g. --with-cxx=clang++, so the fact
              # that it does not work is an immediate problem
              self.mesg += '\n'.join((
                '',
                'Note, you have explicitly requested --{}={}. If you don\'t need {}, or that specific compiler, remove this flag -- configure may be able to find a more suitable compiler automatically.',
                'If you DO need the above, then consult your compilers user manual. It\'s possible you may need to add additional flags (or perhaps load additional modules) to enable compliance'
              )).format(argdbName, argdbVal, LANG.replace('x', '+'))
              raise config.base.ConfigureSetupError(self.mesg)
            self.logPrint(' '.join(('Error testing',LANG,'compiler:',self.mesg)))
            self.delMakeMacro(compilerName)
            delattr(self,compilerName)
          else:
            self.executeTest(getattr(self,LANG.join(('check','Preprocessor'))))
    self.executeTest(self.checkFortranCompiler)
    if hasattr(self, 'FC'):
      self.executeTest(self.checkFortranPreprocessor)
      self.executeTest(self.checkFortranComments)
    self.executeTest(self.checkLargeFileIO)
    self.executeTest(self.checkArchiver)
    self.executeTest(self.checkArchiverRecipeArgfile)
    self.executeTest(self.checkSharedLinker)
    if Configure.isDarwin(self.log):
      self.executeTest(self.checkLinkerMac)
    if Configure.isCygwin(self.log):
      self.executeTest(self.checkLinkerWindows)
    self.executeTest(self.checkPIC)
    self.executeTest(self.checkKandRFlags)
    self.executeTest(self.checkSharedLinkerPaths)
    self.executeTest(self.checkLibC)
    self.executeTest(self.checkDynamicLinker)
    if hasattr(self.framework,'conda_active'):
      del self.framework.additional_error_message

    self.executeTest(self.checkPragma)
    self.executeTest(self.checkAtFileOption)
    self.executeTest(self.output)
    return

  def no_configure(self):
    if self.staticLibraries:
      self.setStaticLinker()
    return
