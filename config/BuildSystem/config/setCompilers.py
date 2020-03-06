from __future__ import generators
import config.base
import config
import os
from functools import reduce

# not sure how to handle this with 'self' so its outside the class
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
                    'void '+export+' foo(void){',
                    '  fprintf_ptr(stdout,"hello");',
                    '  return;',
                    '}',
                    'void bar(void){foo();}\n'])

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.usedMPICompilers = 0
    self.mainLanguage = 'C'
    return

  def __str__(self):
    self.compilerflags = self.framework.getChild('config.compilerFlags')
    desc = ['Compilers:']
    if hasattr(self, 'CC'):
      self.pushLanguage('C')
      desc.append('  C Compiler:         '+self.getCompiler()+' '+self.getCompilerFlags())
      if self.compilerflags.version['C']: desc.append('    Version: '+self.compilerflags.version['C'])
      if not self.getLinker() == self.getCompiler(): desc.append('  C Linker:           '+self.getLinker()+' '+self.getLinkerFlags())
      self.popLanguage()
    if hasattr(self, 'CUDAC'):
      self.pushLanguage('CUDA')
      desc.append('  CUDA Compiler:      '+self.getCompiler()+' '+self.getCompilerFlags())
      if self.compilerflags.version['CUDA']: desc.append('    Version: '+self.compilerflags.version['CUDA'])
      if not self.getLinker() == self.getCompiler(): desc.append('  CUDA Linker:        '+self.getLinker()+' '+self.getLinkerFlags())
      self.popLanguage()
    if hasattr(self, 'CXX'):
      self.pushLanguage('Cxx')
      desc.append('  C++ Compiler:       '+self.getCompiler()+' '+self.getCompilerFlags())
      if self.compilerflags.version['Cxx']: desc.append('    Version: '+self.compilerflags.version['Cxx'])
      if not self.getLinker() == self.getCompiler(): desc.append('  C++ Linker:         '+self.getLinker()+' '+self.getLinkerFlags())
      self.popLanguage()
    if hasattr(self, 'FC'):
      self.pushLanguage('FC')
      desc.append('  Fortran Compiler:   '+self.getCompiler()+' '+self.getCompilerFlags())
      if self.compilerflags.version['FC']: desc.append('    Version: '+self.compilerflags.version['FC'])
      if not self.getLinker() == self.getCompiler(): desc.append('  Fortran Linker:     '+self.getLinker()+' '+self.getLinkerFlags())
      self.popLanguage()
    desc.append('Linkers:')
    if hasattr(self, 'staticLinker'):
      desc.append('  Static linker:   '+self.getSharedLinker()+' '+self.AR_FLAGS)
    elif hasattr(self, 'sharedLinker'):
      desc.append('  Shared linker:   '+self.getSharedLinker()+' '+self.getSharedLinkerFlags())
    if hasattr(self, 'dynamicLinker'):
      desc.append('  Dynamic linker:   '+self.getDynamicLinker()+' '+self.getDynamicLinkerFlags())
      desc.append('  Libraries linked against:   '+self.LIBS)
    return '\n'.join(desc)+'\n'

  def setupHelp(self, help):
    import nargs

    help.addArgument('Compilers', '-with-cpp=<prog>', nargs.Arg(None, None, 'Specify the C preprocessor'))
    help.addArgument('Compilers', '-CPP=<prog>',            nargs.Arg(None, None, 'Specify the C preprocessor'))
    help.addArgument('Compilers', '-CPPFLAGS=<string>',     nargs.Arg(None, None, 'Specify the C only (not used for C++ or FC) preprocessor options'))
    help.addArgument('Compilers', '-with-cc=<prog>',  nargs.Arg(None, None, 'Specify the C compiler'))
    help.addArgument('Compilers', '-CC=<prog>',             nargs.Arg(None, None, 'Specify the C compiler'))
    help.addArgument('Compilers', '-CFLAGS=<string>',       nargs.Arg(None, None, 'Specify the C compiler options'))
    help.addArgument('Compilers', '-CC_LINKER_FLAGS=<string>',        nargs.Arg(None, [], 'Specify the C linker flags'))

    help.addArgument('Compilers', '-CXXPP=<prog>',          nargs.Arg(None, None, 'Specify the C++ preprocessor'))
    help.addArgument('Compilers', '-CXXPPFLAGS=<string>',   nargs.Arg(None, None, 'Specify the C++ preprocessor options'))
    help.addArgument('Compilers', '-with-cxx=<prog>', nargs.Arg(None, None, 'Specify the C++ compiler'))
    help.addArgument('Compilers', '-CXX=<prog>',            nargs.Arg(None, None, 'Specify the C++ compiler'))
    help.addArgument('Compilers', '-CXXFLAGS=<string>',     nargs.Arg(None, None, 'Specify the C++ compiler options, also passed to linker'))
    help.addArgument('Compilers', '-CXX_CXXFLAGS=<string>', nargs.Arg(None, '',   'Specify the C++ compiler-only options, not passed to linker'))
    help.addArgument('Compilers', '-CXX_LINKER_FLAGS=<string>',       nargs.Arg(None, [], 'Specify the C++ linker flags'))

    help.addArgument('Compilers', '-FPP=<prog>',            nargs.Arg(None, None, 'Specify the Fortran preprocessor'))
    help.addArgument('Compilers', '-FPPFLAGS=<string>',     nargs.Arg(None, None, 'Specify the Fortran preprocessor options'))
    help.addArgument('Compilers', '-with-fc=<prog>',  nargs.Arg(None, None, 'Specify the Fortran compiler'))
    help.addArgument('Compilers', '-FC=<prog>',             nargs.Arg(None, None, 'Specify the Fortran compiler'))
    help.addArgument('Compilers', '-FFLAGS=<string>',       nargs.Arg(None, None, 'Specify the Fortran compiler options'))
    help.addArgument('Compilers', '-FC_LINKER_FLAGS=<string>',        nargs.Arg(None, [], 'Specify the FC linker flags'))

    help.addArgument('Compilers', '-with-large-file-io=<bool>', nargs.ArgBool(None, 0, 'Allow IO with files greater then 2 GB'))

    help.addArgument('Compilers', '-CUDAPP=<prog>',        nargs.Arg(None, None, 'Specify the CUDA preprocessor'))
    help.addArgument('Compilers', '-CUDAPPFLAGS=<string>', nargs.Arg(None, '-Wno-deprecated-gpu-targets', 'Specify the CUDA preprocessor options'))
    help.addArgument('Compilers', '-with-cudac=<prog>',    nargs.Arg(None, None, 'Specify the CUDA compiler'))
    help.addArgument('Compilers', '-CUDAC=<prog>',         nargs.Arg(None, None, 'Specify the CUDA compiler'))
    help.addArgument('Compilers', '-CUDAFLAGS=<string>',   nargs.Arg(None, None, 'Specify the CUDA compiler options'))
    help.addArgument('Compilers', '-CUDAC_LINKER_FLAGS=<string>',        nargs.Arg(None, [], 'Specify the CUDA linker flags'))

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
      if output.find('NAGWare Fortran') >= 0 or output.find('The Numerical Algorithms Group Ltd') >= 0:
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isGNU(compiler, log):
    '''Returns true if the compiler is a GNU compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help | head -n 20 ', log = log)
      output = output + error
      return (any([s in output for s in ['www.gnu.org',
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
                                                 ]]))
    except RuntimeError:
      pass

  @staticmethod
  def isClang(compiler, log):
    '''Returns true if the compiler is a Clang/LLVM compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help | head -n 500', log = log, logOutputflg = False)
      output = output + error
      return any([s in output for s in ['Emit Clang AST']])
    except RuntimeError:
      pass

  @staticmethod
  def isGfortran45x(compiler, log):
    '''returns true if the compiler is gfortran-4.5.x'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version', log = log)
      output = output +  error
      import re
      if re.match(r'GNU Fortran \(.*\) (4.5.\d+|4.6.0 20100703)', output):
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isGfortran46plus(compiler, log):
    '''returns true if the compiler is gfortran-4.6.x or later'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version', log = log)
      output = output +  error
      import re
      strmatch = re.match('GNU Fortran\s+\(.*\)\s+(\d+)\.(\d+)',output)
      if strmatch:
        VMAJOR,VMINOR = strmatch.groups()
        if (int(VMAJOR),int(VMINOR)) >= (4,6):
          return 1
    except RuntimeError:
      pass

  @staticmethod
  def isGfortran47plus(compiler, log):
    '''returns true if the compiler is gfortran-4.7.x or later'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version', log = log)
      output = output +  error
      import re
      strmatch = re.match('GNU Fortran\s+\(.*\)\s+(\d+)\.(\d+)',output)
      if strmatch:
        VMAJOR,VMINOR = strmatch.groups()
        if (int(VMAJOR),int(VMINOR)) >= (4,7):
          return 1
    except RuntimeError:
      pass

  @staticmethod
  def isGfortran8plus(compiler, log):
    '''returns true if the compiler is gfortran-8 or later'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version', log = log)
      output = output +  error
      import re
      strmatch = re.match('GNU Fortran\s+\(.*\)\s+(\d+)\.(\d+)',output)
      if strmatch:
        VMAJOR,VMINOR = strmatch.groups()
        if (int(VMAJOR),int(VMINOR)) >= (8,0):
          return 1
    except RuntimeError:
      pass

  @staticmethod
  def isG95(compiler, log):
    '''Returns true if the compiler is g95'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help | head -n 20', log = log)
      output = output + error
      if output.find('Unrecognised option --help passed to ld') >=0:    # NAG f95 compiler
        return 0
      if output.find('http://www.g95.org') >= 0:
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isCompaqF90(compiler, log):
    '''Returns true if the compiler is Compaq f90'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help | head -n 20', log = log)
      output = output + error
      if output.find('Unrecognised option --help passed to ld') >=0:    # NAG f95 compiler
        return 0
      if output.find('Compaq Visual Fortran') >= 0 or output.find('Digital Visual Fortran') >=0 :
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isSun(compiler, log):
    '''Returns true if the compiler is a Sun/Oracle compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' -V',checkCommand = noCheck, log = log)
      output = output + error
      if output.find(' Sun ') >= 0:
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
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isIntel(compiler, log):
    '''Returns true if the compiler is a Intel compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help | head -n 20', log = log)
      output = output + error
      if output.find('Intel Corporation') >= 0 :
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isCrayKNL(compiler, log):
    '''Returns true if the compiler is a compiler for KNL running on a Cray'''
    x = os.getenv('PE_PRODUCT_LIST')
    if x and x.find('CRAYPE_MIC-KNL') > -1:
      return 1

  @staticmethod
  def isCray(compiler, log):
    '''Returns true if the compiler is a Cray compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' -V', log = log)
      output = output + error
      if output.find('Cray Standard C') >= 0 or output.find('Cray C++') >= 0 or output.find('Cray Fortran') >= 0:
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
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isPGI(compiler, log):
    '''Returns true if the compiler is a PGI compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' -V',checkCommand = noCheck, log = log)
      output = output + error
      if output.find('The Portland Group') >= 0:
        return 1
    except RuntimeError:
      pass

  @staticmethod
  def isSolarisAR(ar, log):
    '''Returns true AR is solaris'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(ar + ' -V',checkCommand = noCheck, log = log)
      output = output + error
      if output.find('Software Generation Utilities') >= 0:
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
  def isLinux(log):
    '''Returns true if system is linux'''
    (output, error, status) = config.base.Configure.executeShellCommand('uname -s', log = log)
    if not status and output.lower().strip().find('linux') >= 0:
      return 1

  @staticmethod
  def isCygwin(log):
    '''Returns true if system is cygwin'''
    (output, error, status) = config.base.Configure.executeShellCommand('uname -s', log = log)
    if not status and output.lower().strip().find('cygwin') >= 0:
      return 1

  @staticmethod
  def isSolaris(log):
    '''Returns true if system is solaris'''
    (output, error, status) = config.base.Configure.executeShellCommand('uname -s', log = log)
    if not status and output.lower().strip().find('sunos') >= 0:
      return 1

  @staticmethod
  def isDarwin(log):
    '''Returns true if system is Darwin/MacOSX'''
    (output, error, status) = config.base.Configure.executeShellCommand('uname -s', log = log)
    if not status:
      return output.lower().strip() == 'darwin'

  @staticmethod
  def isDarwinCatalina(log):
    '''Returns true if system is Darwin/MacOSX Version Catalina or higher'''
    import platform
    if platform.system() != 'Darwin': return 0
    v = tuple([int(a) for a in platform.mac_ver()[0].split('.')])
    if v < (10,15,0): return 0
    return 1

  @staticmethod
  def isFreeBSD(log):
    '''Returns true if system is FreeBSD'''
    (output, error, status) = config.base.Configure.executeShellCommand('uname -s', log = log)
    if not status:
      return output.lower().strip() == 'freebsd'

  @staticmethod
  def isWindows(compiler, log):
    '''Returns true if the compiler is a Windows compiler'''
    if compiler in ['icl', 'cl', 'bcc32', 'ifl', 'df']:
      return 1
    if compiler in ['ifort','f90'] and Configure.isCygwin(log):
      return 1
    if compiler in ['lib', 'tlib']:
      return 1

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
    if 'with-mpi' in self.argDB and self.argDB['with-mpi'] and self.argDB['with-mpi-compilers']:
      return 1
    return 0

  def checkInitialFlags(self):
    '''Initialize the compiler and linker flags'''
    for language in ['C', 'CUDA', 'Cxx', 'FC']:
      self.pushLanguage(language)
      for flagsArg in [config.base.Configure.getCompilerFlagsName(language), config.base.Configure.getCompilerFlagsName(language, 1), config.base.Configure.getLinkerFlagsName(language)]:
        if flagsArg in self.argDB: setattr(self, flagsArg, self.argDB[flagsArg])
        else: setattr(self, flagsArg, '')
        self.logPrint('Initialized '+flagsArg+' to '+str(getattr(self, flagsArg)))
      self.popLanguage()
    for flagsArg in ['CPPFLAGS', 'FPPFLAGS', 'CUDAPPFLAGS', 'CXXPPFLAGS', 'CC_LINKER_FLAGS', 'CXX_LINKER_FLAGS', 'FC_LINKER_FLAGS', 'CUDAC_LINKER_FLAGS','sharedLibraryFlags', 'dynamicLibraryFlags']:
      if flagsArg in self.argDB: setattr(self, flagsArg, self.argDB[flagsArg])
      else: setattr(self, flagsArg, '')
      self.logPrint('Initialized '+flagsArg+' to '+str(getattr(self, flagsArg)))
    if 'LIBS' in self.argDB:
      self.LIBS = self.argDB['LIBS']
    else:
      self.LIBS = ''
    return

  def checkCompiler(self, language, linkLanguage=None,includes = '', body = '', cleanup = 1, codeBegin = None, codeEnd = None):
    '''Check that the given compiler is functional, and if not raise an exception'''
    self.pushLanguage(language)
    if not self.checkCompile(includes, body, cleanup, codeBegin, codeEnd):
      msg = 'Cannot compile '+language+' with '+self.getCompiler()+'.'
      self.popLanguage()
      raise RuntimeError(msg)
    if language == 'CUDA': # do not check CUDA linker since it is never used (and is broken on Mac with -m64)
      self.popLanguage()
      return
    if not self.checkLink(linkLanguage=linkLanguage,includes=includes,body=body):
      msg = 'Cannot compile/link '+language+' with '+self.getCompiler()+'.'
      self.popLanguage()
      raise RuntimeError(msg)
    oldlibs = self.LIBS
    self.LIBS += ' -lpetsc-ufod4vtr9mqHvKIQiVAm'
    if self.checkLink(linkLanguage=linkLanguage):
      msg = language + ' compiler ' + self.getCompiler()+ ''' is broken! It is returning a zero error when the linking failed! Either
 1) switch to another compiler suite or
 2) report this entire error message to your compiler/linker suite vendor and ask for fix for this issue.'''
      self.popLanguage()
      self.LIBS = oldlibs
      raise RuntimeError(msg)
    self.LIBS = oldlibs
    if not self.argDB['with-batch']:
      if not self.checkRun(linkLanguage=linkLanguage):
        msg = 'Cannot run executables created with '+language+'. If this machine uses a batch system \nto submit jobs you will need to configure using ./configure with the additional option  --with-batch.\n Otherwise there is problem with the compilers. Can you compile and run code with your compiler \''+ self.getCompiler()+'\'?\n'
        if self.isIntel(self.getCompiler(), self.log):
          msg = msg + 'See https://www.mcs.anl.gov/petsc/documentation/faq.html#libimf'
        self.popLanguage()
        raise OSError(msg)
    self.popLanguage()
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
      elif self.argDB['download-openmpi']: mesg ='with downloaded OpenMPI'
      else: mesg = ''
      raise RuntimeError('Error '+mesg+': '+self.mesg)
    elif 'with-cc' in self.argDB:
      if self.isWindows(self.argDB['with-cc'], self.log):
        yield 'win32fe '+self.argDB['with-cc']
      else:
        yield self.argDB['with-cc']
      raise RuntimeError('C compiler you provided with -with-cc='+self.argDB['with-cc']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif 'CC' in self.argDB:
      if self.isWindows(self.argDB['CC'], self.log):
        yield 'win32fe '+self.argDB['CC']
      else:
        yield self.argDB['CC']
      raise RuntimeError('C compiler you provided with -CC='+self.argDB['CC']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif self.useMPICompilers() and 'with-mpi-dir' in self.argDB and os.path.isdir(os.path.join(self.argDB['with-mpi-dir'], 'bin')):
      self.usedMPICompilers = 1
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpiicc')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpicc')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpcc')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'hcc')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpcc_r')
      self.usedMPICompilers = 0
      raise RuntimeError('MPI compiler wrappers in '+self.argDB['with-mpi-dir']+'/bin cannot be found or do not work. See https://www.mcs.anl.gov/petsc/documentation/faq.html#mpi-compilers')
    else:
      if self.useMPICompilers() and 'with-mpi-dir' in self.argDB:
      # if it gets here these means that self.argDB['with-mpi-dir']/bin does not exist so we should not search for MPI compilers
      # that is we are turning off the self.useMPICompilers()
        self.logPrintBox('***** WARNING: '+os.path.join(self.argDB['with-mpi-dir'], 'bin')+ ' dir does not exist!\n Skipping check for MPI compilers due to potentially incorrect --with-mpi-dir option.\n Suggest using --with-cc=/path/to/mpicc option instead ******')

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
        yield 'mpicc'
        yield 'mpiicc'
        yield 'mpcc_r'
        yield 'mpcc'
        yield 'mpxlc'
        yield 'hcc'
        self.usedMPICompilers = 0
      yield 'gcc'
      yield 'clang'
      yield 'icc'
      yield 'cc'
      yield 'xlc'
      yield 'win32fe icl'
      yield 'win32fe cl'
      yield 'pgcc'
      yield 'win32fe bcc32'
    return

  def checkCCompiler(self):
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
        if os.path.basename(self.CC) == 'mpicc':
          self.logPrint(' MPI installation '+str(self.CC)+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI.')
        self.delMakeMacro('CC')
        del self.CC
    if not hasattr(self, 'CC'):
      raise RuntimeError('Could not locate a functional C compiler')
    try:
      self.executeShellCommand(self.CC+' --version', log = self.log)
    except:
      pass
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
    for compiler in self.generateCPreprocessorGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'CPP'):
          self.pushLanguage('C')
          if not self.checkPreprocess('#include <stdlib.h>\n'):
            raise RuntimeError('Cannot preprocess C with '+self.CPP+'.')
          self.popLanguage()
          return
      except RuntimeError as e:
        self.popLanguage()
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
    return

  def checkCUDACompiler(self):
    '''Locate a functional CUDA compiler'''
    if ('with-cudac' in self.argDB and self.argDB['with-cudac'] == '0'):
      if 'CUDAC' in self.argDB:
        del self.argDB['CUDAC']
      return
    self.mesg = ''
    for compiler in self.generateCUDACompilerGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'CUDAC'):
          self.checkCompiler('CUDA')
          # Put version info into the log
          compilerVersion = self.executeShellCommand(self.CUDAC+' --version', log = self.log)
          compilerVersion = compilerVersion[0]
          compilerVersion = compilerVersion.split()
          i = 0
          for word in compilerVersion:
            i = i+1
            if word == 'release':
              break
          self.compilerVersionCUDA = compilerVersion[i].strip(',')
          break
      except RuntimeError as e:
        self.mesg = str(e)
        self.logPrint('Error testing CUDA compiler: '+str(e))
        self.delMakeMacro('CUDAC')
        del self.CUDAC
    return

  def generateCUDAPreprocessorGuesses(self):
    '''Determines the CUDA preprocessor from --with-cudacpp, then CUDAPP, then the CUDA compiler'''
    if 'with-cudacpp' in self.argDB:
      yield self.argDB['with-cudacpp']
    elif 'CUDAPP' in self.argDB:
      yield self.argDB['CUDAPP']
    else:
      if hasattr(self, 'CUDAC'):
        yield self.CUDAC+' -E'
    return

  def checkCUDAPreprocessor(self):
    '''Locate a functional CUDA preprocessor'''
    for compiler in self.generateCUDAPreprocessorGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'CUDAPP'):
          self.pushLanguage('CUDA')
          if not self.checkPreprocess('#include <stdlib.h>\n__global__ void testFunction() {return;};'):
            raise RuntimeError('Cannot preprocess CUDA with '+self.CUDAPP+'.')
          self.popLanguage()
          return
      except RuntimeError as e:
        self.popLanguage()
    return

  def generateCxxCompilerGuesses(self):
    '''Determine the Cxx compiler'''

    if hasattr(self, 'CXX'):
      yield self.CXX
      if self.argDB['download-mpich']: mesg ='with downloaded MPICH'
      elif self.argDB['download-openmpi']: mesg ='with downloaded OpenMPI'
      else: mesg = ''
      raise RuntimeError('Error '+mesg+': '+self.mesg)
    elif 'with-c++' in self.argDB:
      raise RuntimeError('Keyword --with-c++ is WRONG, use --with-cxx')
    if 'with-CC' in self.argDB:
      raise RuntimeError('Keyword --with-CC is WRONG, use --with-cxx')

    if 'with-cxx' in self.argDB:
      if self.argDB['with-cxx'] == 'gcc': raise RuntimeError('Cannot use C compiler gcc as the C++ compiler passed in with --with-cxx')
      if self.isWindows(self.argDB['with-cxx'], self.log):
        yield 'win32fe '+self.argDB['with-cxx']
      else:
        yield self.argDB['with-cxx']
      raise RuntimeError('C++ compiler you provided with -with-cxx='+self.argDB['with-cxx']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif 'CXX' in self.argDB:
      if self.isWindows(self.argDB['CXX'], self.log):
        yield 'win32fe '+self.argDB['CXX']
      else:
        yield self.argDB['CXX']
      raise RuntimeError('C++ compiler you provided with -CXX='+self.argDB['CXX']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif self.useMPICompilers() and 'with-mpi-dir' in self.argDB and os.path.isdir(os.path.join(self.argDB['with-mpi-dir'], 'bin')):
      self.usedMPICompilers = 1
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpiicpc')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpicxx')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'hcp')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpic++')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpiCC')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpCC_r')
      self.usedMPICompilers = 0
      raise RuntimeError('bin/<mpiCC,mpicxx,hcp,mpCC_r> you provided with -with-mpi-dir='+self.argDB['with-mpi-dir']+' cannot be found or does not work. See https://www.mcs.anl.gov/petsc/documentation/faq.html#mpi-compilers')
    else:
      if self.useMPICompilers():
        self.usedMPICompilers = 1
        cray = os.getenv('CRAYPE_DIR')
        if cray:
          cross_CC = self.crayCrossCompiler('CC')
          if cross_CC:
            self.cross_CC = cross_CC
            self.log.write('Cray system using C++ cross compiler:'+cross_CC+'\n')
          yield 'CC'
          if cross_CC: delattr(self, 'cross_CC')
        yield 'mpicxx'
        yield 'mpiicpc'
        yield 'mpCC_r'
        yield 'mpiCC'
        yield 'mpic++'
        yield 'mpCC'
        yield 'mpxlC'
        self.usedMPICompilers = 0
      #attempt to match c++ compiler with c compiler
      if self.CC.find('win32fe cl') >= 0:
        yield 'win32fe cl'
      elif self.CC.find('win32fe icl') >= 0:
        yield 'win32fe icl'
      elif self.CC == 'gcc':
        yield 'g++'
      elif self.CC == 'clang':
        yield 'clang++'
      elif self.CC == 'icc':
        yield 'icpc'
      elif self.CC == 'xlc':
        yield 'xlC'
      yield 'g++'
      yield 'clang++'
      yield 'c++'
      yield 'icpc'
      yield 'CC'
      yield 'cxx'
      yield 'cc++'
      yield 'xlC'
      yield 'ccpc'
      yield 'win32fe icl'
      yield 'win32fe cl'
      yield 'pgCC'
      yield 'CC'
      yield 'win32fe bcc32'
    return

  def checkCxxCompiler(self):
    '''Locate a functional Cxx compiler'''
    if 'with-cxx' in self.argDB and self.argDB['with-cxx'] == '0':
      if 'CXX' in self.argDB:
        del self.argDB['CXX']
      return
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
          if os.path.basename(self.CXX) in ['mpicxx', 'mpiCC']:
            self.logPrint('  MPI installation '+str(self.CXX)+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI.')
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
    if not hasattr(self, 'CXX'):
      return
    for compiler in self.generateCxxPreprocessorGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'CXXPP'):
          self.pushLanguage('Cxx')
          if not self.checkPreprocess('#include <cstdlib>\n'):
            raise RuntimeError('Cannot preprocess Cxx with '+self.CXXPP+'.')
          self.popLanguage()
          break
      except RuntimeError as e:

        if os.path.basename(self.CXXPP) in ['mpicxx', 'mpiCC']:
          self.logPrint('MPI installation '+self.getCompiler()+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI')
        self.popLanguage()
        self.delMakeMacro('CXXPP')
        del self.CXXPP
    return

  def generateFortranCompilerGuesses(self):
    '''Determine the Fortran compiler'''

    if hasattr(self, 'FC'):
      yield self.FC
      if self.argDB['download-mpich']: mesg ='with downloaded MPICH'
      elif self.argDB['download-openmpi']: mesg ='with downloaded OpenMPI'
      else: mesg = ''
      raise RuntimeError('Error '+mesg+': '+self.mesg)
    elif 'with-fc' in self.argDB:
      if self.isWindows(self.argDB['with-fc'], self.log):
        yield 'win32fe '+self.argDB['with-fc']
      else:
        yield self.argDB['with-fc']
      raise RuntimeError('Fortran compiler you provided with --with-fc='+self.argDB['with-fc']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif 'FC' in self.argDB:
      if self.isWindows(self.argDB['FC'], self.log):
        yield 'win32fe '+self.argDB['FC']
      else:
        yield self.argDB['FC']
      yield self.argDB['FC']
      raise RuntimeError('Fortran compiler you provided with -FC='+self.argDB['FC']+' cannot be found or does not work.'+'\n'+self.mesg)
    elif self.useMPICompilers() and 'with-mpi-dir' in self.argDB and os.path.isdir(os.path.join(self.argDB['with-mpi-dir'], 'bin')):
      self.usedMPICompilers = 1
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpiifort')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpif90')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpf90')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpxlf95_r')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpxlf90_r')
      yield os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpxlf_r')
      self.usedMPICompilers = 0
      if os.path.isfile(os.path.join(self.argDB['with-mpi-dir'], 'bin', 'mpif90')):
        raise RuntimeError('bin/mpif90 you provided with --with-mpi-dir='+self.argDB['with-mpi-dir']+' cannot be found or does not work.\nRun with --with-fc=0 if you wish to use this MPI and disable Fortran. See https://www.mcs.anl.gov/petsc/documentation/faq.html#mpi-compilers')
    else:
      if self.useMPICompilers():
        self.usedMPICompilers = 1
        cray = os.getenv('CRAYPE_DIR')
        if cray:
          cross_fc = self.crayCrossCompiler('ftn')
          if cross_fc:
            self.cross_fc = cross_fc
            self.log.write('Cray system using Fortran cross compiler:'+cross_fc+'\n')
          yield 'ftn'
          if cross_fc: delattr(self, 'cross_fc')
        yield 'mpif90'
        yield 'mpiifort'
        yield 'mpxlf_r'
        yield 'mpxlf'
        yield 'mpf90'
        self.usedMPICompilers = 0
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
      elif self.CC.find('win32fe cl') >= 0:
        yield 'win32fe f90'
        yield 'win32fe ifc'
      elif self.CC.find('win32fe icl') >= 0:
        yield 'win32fe ifc'
      yield 'gfortran'
      yield 'g95'
      yield 'xlf90'
      yield 'xlf'
      yield 'f90'
      yield 'lf95'
      yield 'win32fe ifort'
      yield 'win32fe ifl'
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
        if os.path.basename(self.FC) in ['mpif90']:
          self.logPrint(' MPI installation '+str(self.FC)+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI.')
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
    for compiler in self.generateFortranPreprocessorGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'FPP'):
          self.pushLanguage('FC')
          if not self.checkPreprocess('#define foo 10\n'):
            raise RuntimeError('Cannot preprocess Fortran with '+self.FPP+'.')
          self.popLanguage()
          break
      except RuntimeError as e:

        if os.path.basename(self.FPP) in ['mpif90']:
          self.logPrint('MPI installation '+self.getCompiler()+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI')
        self.popLanguage()
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
    if (output.find('Unrecognized command line option') >= 0 or output.find('Unrecognised command line option') >= 0 or
        output.find('unrecognized command line option') >= 0 or output.find('unrecognized option') >= 0 or output.find('unrecognised option') >= 0 or
        output.find('not recognized') >= 0 or output.find('not recognised') >= 0 or
        output.find('unknown option') >= 0 or output.find('unknown flag') >= 0 or output.find('Unknown switch') >= 0 or
        output.find('ignoring option') >= 0 or output.find('ignored') >= 0 or
        output.find('argument unused') >= 0 or output.find('not supported') >= 0 or
        # When checking for the existence of 'attribute'
        output.find('is unsupported and will be skipped') >= 0 or
        output.find('illegal option') >= 0 or output.find('Invalid option') >= 0 or
        (output.find('bad ') >= 0 and output.find(' option') >= 0) or
        output.find('linker input file unused because linking not done') >= 0 or
        output.find('PETSc Error') >= 0 or
        output.find('Unbekannte Option') >= 0 or
        output.find('warning: // comments are not allowed in this language') >= 0 or
        output.find('no se reconoce la opci') >= 0) or output.find('non reconnue') >= 0:
      return 1
    return 0

  def checkCompilerFlag(self, flag, includes = '', body = '', compilerOnly = 0):
    '''Determine whether the compiler accepts the given flag'''
    flagsArg = self.getCompilerFlagsArg(compilerOnly)
    oldFlags = getattr(self, flagsArg)
    setattr(self, flagsArg, oldFlags+' '+flag)
    (output, error, status) = self.outputCompile(includes, body)
    output += error
    valid   = 1
    setattr(self, flagsArg, oldFlags)
    # Please comment each entry and provide an example line
    if status:
      valid = 0
      self.logPrint('Rejecting compiler flag '+flag+' due to nonzero status from link')
    # Lahaye F95
    if output.find('Invalid suboption') >= 0:
      valid = 0
    if self.containsInvalidFlag(output):
      valid = 0
      self.logPrint('Rejecting compiler flag '+flag+' due to \n'+output)
    return valid

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

  def generatePICGuesses(self):
    yield ''
    if self.language[-1] == 'CUDA':
      yield '-Xcompiler -fPIC'
    elif config.setCompilers.Configure.isGNU(self.getCompiler(), self.log):
      yield '-fPIC'
    else:
      yield '-PIC'
      yield '-fPIC'
      yield '-KPIC'
      yield '-qpic'
    return

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
      raise RuntimeError("Cannot determine compiler PIC flags if shared libraries is turned off\nEither run using --with-shared-libraries or --with-pic=0 and supply the compiler PIC flag via CFLAGS, CXXXFLAGS, and FCFLAGS\n")
    if self.sharedLibraries and self.mainLanguage == 'C': languages = []
    else: languages = ['C']
    if hasattr(self, 'CXX'):
      languages.append('Cxx')
    if hasattr(self, 'FC'):
      languages.append('FC')
    if hasattr(self, 'CUDAC'):
      languages.append('CUDA')
    for language in languages:
      self.pushLanguage(language)
      if language in ['C','Cxx','CUDA']:
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

  def checkLargeFileIO(self):
    # check for large file support with 64bit offset
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
    elif prog == 'win32fe':
      args = os.path.basename(archiver).split(' ')
      if 'lib' in args:
        flag = '-a'
      elif 'tlib' in args:
        flag = '-a -P512'
    if prog.endswith('ar') and not (self.isSolarisAR(prog, self.log) or self.isAIXAR(prog, self.log)):
      self.FAST_AR_FLAGS = 'Scq'
    else:
      self.FAST_AR_FLAGS = flag
    self.framework.addMakeMacro('FAST_AR_FLAGS',self.FAST_AR_FLAGS )
    return flag

  def generateArchiverGuesses(self):
    defaultAr = None
    if 'with-ar' in self.argDB:
      if self.isWindows(self.argDB['with-ar'], self.log):
        defaultAr = 'win32fe '+self.argDB['with-ar']
      else:
        defaultAr = self.argDB['with-ar']
    envAr = None
    if 'AR' in self.argDB:
      if self.isWindows(self.argDB['AR'], self.log):
        envAr = 'win32fe '+self.argDB['AR']
      else:
        envAr = self.argDB['AR']
    defaultRanlib = None
    if 'with-ranlib' in self.argDB:
      defaultRanlib = self.argDB['with-ranlib']
    envRanlib = None
    if 'RANLIB' in self.argDB:
      envRanlib = self.argDB['RANLIB']
    if defaultAr and defaultRanlib:
      yield(defaultAr,self.getArchiverFlags(defaultAr),defaultRanlib)
      raise RuntimeError('The archiver set --with-ar="'+defaultAr+'" is incompatible with the ranlib set --with-ranlib="'+defaultRanlib+'".')
    if defaultAr and envRanlib:
      yield(defaultAr,self.getArchiverFlags(defaultAr),envRanlib)
      raise RuntimeError('The archiver set --with-ar="'+defaultAr+'" is incompatible with the ranlib set (perhaps in your environment) -RANLIB="'+envRanlib+'".')
    if envAr and defaultRanlib:
      yield(envAr,self.getArchiverFlags(envAr),defaultRanlib)
      raise RuntimeError('The archiver set --AR="'+envAr+'" is incompatible with the ranlib set --with-ranlib="'+defaultRanlib+'".')
    if envAr and envRanlib:
      yield(envAr,self.getArchiverFlags(envAr),envRanlib)
      raise RuntimeError('The archiver set --AR="'+envAr+'" is incompatible with the ranlib set (perhaps in your environment) -RANLIB="'+envRanlib+'".')
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
      yield ('win32fe tlib',self.getArchiverFlags('win32fe tlib'),defaultRanlib)
      yield ('win32fe lib',self.getArchiverFlags('win32fe lib'),defaultRanlib)
      raise RuntimeError('You set --with-ranlib="'+defaultRanlib+'", but '+defaultRanlib+' cannot be used\n')
    if envRanlib:
      yield ('ar',self.getArchiverFlags('ar'),envRanlib)
      yield ('win32fe tlib',self.getArchiverFlags('win32fe tlib'),envRanlib)
      yield ('win32fe lib',self.getArchiverFlags('win32fe lib'),envRanlib)
      raise RuntimeError('You set -RANLIB="'+envRanlib+'" (perhaps in your environment), but '+defaultRanlib+' cannot be used\n')
    yield ('ar',self.getArchiverFlags('ar'),'ranlib -c')
    yield ('ar',self.getArchiverFlags('ar'),'ranlib')
    yield ('ar',self.getArchiverFlags('ar'),'true')
    # IBM with 64 bit pointers
    yield ('ar','-X64 '+self.getArchiverFlags('ar'),'ranlib -c')
    yield ('ar','-X64 '+self.getArchiverFlags('ar'),'ranlib')
    yield ('ar','-X64 '+self.getArchiverFlags('ar'),'true')
    yield ('win32fe tlib',self.getArchiverFlags('win32fe tlib'),'true')
    yield ('win32fe lib',self.getArchiverFlags('win32fe lib'),'true')
    return

  def checkArchiver(self):
    '''Check that the archiver exists and can make a library usable by the compiler'''
    objName    = os.path.join(self.tmpDir, 'conf1.o')
    arcUnix    = os.path.join(self.tmpDir, 'libconf1.a')
    arcWindows = os.path.join(self.tmpDir, 'libconf1.lib')
    def checkArchive(command, status, output, error):
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
          success =  self.checkLink('extern int foo(int);', '  int b = foo(1);  if (b);\n')
          os.rename(arcUnix, arcWindows)
          if not success:
            arext = 'lib'
            success = self.checkLink('extern int foo(int);', '  int b = foo(1);  if (b);\n')
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
      raise RuntimeError('Could not find a suitable archiver.  Use --with-ar to specify an archiver.')
    self.AR_FLAGS      = arflags
    self.AR_LIB_SUFFIX = arext
    self.framework.addMakeMacro('AR_FLAGS', self.AR_FLAGS)
    self.addMakeMacro('AR_LIB_SUFFIX', self.AR_LIB_SUFFIX)
    os.remove(objName)
    self.LIBS = oldLibs
    self.popLanguage()
    return

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
        yield (self.argDB['with-shared-ld'], ['-dynamiclib -single_module', '-undefined dynamic_lookup', '-multiply_defined suppress', '-no_compact_unwind'], 'dylib')
      if hasattr(self, 'CXX') and self.mainLanguage == 'Cxx':
        yield (self.CXX, ['-dynamiclib -single_module', '-undefined dynamic_lookup', '-multiply_defined suppress', '-no_compact_unwind'], 'dylib')
      yield (self.CC, ['-dynamiclib -single_module', '-undefined dynamic_lookup', '-multiply_defined suppress', '-no_compact_unwind'], 'dylib')
    if hasattr(self, 'CXX') and self.mainLanguage == 'Cxx':
      # C++ compiler default
      yield (self.CXX, ['-shared'], 'so')
      yield (self.CXX, ['-dynamic'], 'so')
    # C compiler default
    yield (self.CC, ['-shared'], 'so')
    yield (self.CC, ['-dynamic'], 'so')
    yield (self.CC, ['-qmkshrobj'], 'so')
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
        del self.sharedLinker
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
    if self.containsInvalidFlag(output):
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
    langMap = {'C':'CC','FC':'FC','Cxx':'CXX','CUDA':'CUDAC'}
    languages = ['C']
    if hasattr(self, 'CXX'):
      languages.append('Cxx')
    if hasattr(self, 'FC'):
      languages.append('FC')
    for language in languages:
      self.pushLanguage(language)
      for testFlag in ['-Wl,-multiply_defined,suppress', '-Wl,-multiply_defined -Wl,suppress', '-Wl,-commons,use_dylibs', '-Wl,-search_paths_first', '-Wl,-no_compact_unwind']:
        if self.checkLinkerFlag(testFlag):
          # expand to CC_LINKER_FLAGS or CXX_LINKER_FLAGS or FC_LINKER_FLAGS
          linker_flag_var = langMap[language]+'_LINKER_FLAGS'
          val = getattr(self,linker_flag_var)
          val.append(testFlag)
          setattr(self,linker_flag_var,val)
      self.popLanguage()
    return

  def checkLinkerWindows(self):
    '''Turns off linker warning about unknown .o files extension'''
    langMap = {'C':'CC','FC':'FC','Cxx':'CXX','CUDA':'CUDAC'}
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
    for language in languages:
      flag = '-L'
      self.pushLanguage(language)
      # test '-R' before '-rpath' as sun compilers [c,fortran] don't give proper errors with wrong options.
      if not Configure.isDarwin(self.log):
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
    self.logPrint('*** WARNING *** Shared linking may not function on this architecture')
    self.staticLibrary=1
    self.sharedLibrary=0

  def generateDynamicLinkerGuesses(self):
    if 'with-dynamic-ld' in self.argDB:
      yield (self.argDB['with-dynamic-ld'], [], 'so')
    # Mac OSX
    if Configure.isDarwin(self.log):
      if 'with-dynamic-ld' in self.argDB:
        yield (self.argDB['with-dynamic-ld'], ['-dynamiclib -single_module -undefined dynamic_lookup -multiply_defined suppress'], 'dylib')
      if hasattr(self, 'CXX') and self.mainLanguage == 'Cxx':
        yield (self.CXX, ['-dynamiclib -single_module -undefined dynamic_lookup -multiply_defined suppress'], 'dylib')
      yield (self.CC, ['-dynamiclib -single_module -undefined dynamic_lookup -multiply_defined suppress'], 'dylib')
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
    '''Check that the linker can dynamicaly load shared libraries'''
    self.dynamicLibraries = 0
    if not self.headers.check('dlfcn.h'):
      self.logPrint('Dynamic loading disabled since dlfcn.h was missing')
      return
    self.libraries.saveLog()
    if not self.libraries.add('dl', ['dlopen', 'dlsym', 'dlclose']):
      if not self.libraries.check('', ['dlopen', 'dlsym', 'dlclose']):
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
      optcplrs = [(['with-cc','CC'],['mpiicc','mpicc','mpcc','hcc','mpcc_r']),
              (['with-fc','FC'],['mpiifort','mpif90','mpxlf95_r','mpxlf90_r','mpxlf_r','mpf90']),
              (['with-cxx','CXX'],['mpiicpc','mpicxx','hcp','mpic++','mpiCC','mpCC_r'])]
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
    '''OpenMPI wrappers require LD_LIBRARY_PATH set'''
    if 'with-mpi-dir' in self.argDB:
      libdir = os.path.join(self.argDB['with-mpi-dir'], 'lib')
      if os.path.exists(os.path.join(libdir,'libopen-rte.so')):
        Configure.addLdPath(libdir)
        self.logPrint('Adding to LD_LIBRARY_PATH '+libdir)
    return

  def printEnvVariables(self):
    buf = '**** printenv ****'
    for key,val in os.environ.items():
      buf += '\n'+str(key)+'='+str(val)
    self.logPrint(buf)
    return

  def resetEnvCompilers(self):
    ignoreEnvCompilers = ['CC','CXX','FC','F77','F90']
    for envVal in ignoreEnvCompilers:
      if envVal in os.environ:
        if envVal in self.framework.clArgDB or 'with-'+envVal.lower() in self.framework.clArgDB:
          self.logPrint(envVal+' (set to '+os.environ[envVal]+') found in environment variables - ignoring since also set on command line')
          del os.environ[envVal]
        elif self.argDB['with-environment-variables']:
          self.logPrintBox('***** WARNING: '+envVal+' (set to '+os.environ[envVal]+') found in environment variables - using it \n use ./configure --disable-environment-variables to NOT use the environmental variables ******')
        elif self.framework.argDB['with-xsdk-defaults'] and 'with-environment-variables' not in self.framework.clArgDB:
          self.logPrintBox('***** WARNING: '+envVal+' (set to '+os.environ[envVal]+') found in environment variables - using it \n because --with-xsdk-defaults was selected. Add --disable-environment-variables \n to NOT use the environmental variables ******')

        else:
          self.logPrintBox('***** WARNING: '+envVal+' (set to '+os.environ[envVal]+') found in environment variables - ignoring \n use ./configure '+envVal+'=$'+envVal+' if you really want to use that value ******')
          del os.environ[envVal]

    ignoreEnv = ['CFLAGS','CXXFLAGS','FCFLAGS','FFLAGS','F90FLAGS','CPP','CPPFLAGS','CXXPP','CXXPPFLAGS','LDFLAGS','LIBS','MPI_DIR','RM','MAKEFLAGS','AR']
    for envVal in ignoreEnv:
      if envVal in os.environ:
        if envVal in self.framework.clArgDB:
          self.logPrint(envVal+' (set to '+os.environ[envVal]+') found in environment variables - ignoring since also set on command line')
          del os.environ[envVal]
        elif self.argDB['with-environment-variables']:
          self.logPrintBox('***** WARNING: '+envVal+' (set to '+os.environ[envVal]+') found in environment variables - using it \n use ./configure --disable-environment-variables to NOT use the environmental variables******')
        else:
          self.logPrintBox('***** WARNING: '+envVal+' (set to '+os.environ[envVal]+') found in environment variables - ignoring \n use ./configure '+envVal+'=$'+envVal+' if you really want to use that value ******')
          del os.environ[envVal]
    return


  def checkEnvCompilers(self):
    if 'with-environment-variables' in self.framework.clArgDB or 'with-xsdk-defaults' in self.framework.clArgDB:
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

  def configure(self):
    self.mainLanguage = self.languages.clanguage
    self.executeTest(self.printEnvVariables)
    self.executeTest(self.resetEnvCompilers)
    self.executeTest(self.checkEnvCompilers)
    self.executeTest(self.checkMPICompilerOverride)
    self.executeTest(self.requireMpiLdPath)
    self.executeTest(self.checkInitialFlags)
    self.executeTest(self.checkCCompiler)
    self.executeTest(self.checkCPreprocessor)
    self.executeTest(self.checkCUDACompiler)
    self.executeTest(self.checkCUDAPreprocessor)
    self.executeTest(self.checkCxxCompiler)
    if hasattr(self, 'CXX'):
      self.executeTest(self.checkCxxPreprocessor)
    self.executeTest(self.checkFortranCompiler)
    if hasattr(self, 'FC'):
      self.executeTest(self.checkFortranPreprocessor)
      self.executeTest(self.checkFortranComments)
    self.executeTest(self.checkLargeFileIO)
    self.executeTest(self.checkArchiver)
    self.executeTest(self.checkSharedLinker)
    if Configure.isDarwin(self.log):
      self.executeTest(self.checkLinkerMac)
    if Configure.isCygwin(self.log):
      self.executeTest(self.checkLinkerWindows)
    self.executeTest(self.checkPIC)
    self.executeTest(self.checkSharedLinkerPaths)
    self.executeTest(self.checkLibC)
    self.executeTest(self.checkDynamicLinker)
    self.executeTest(self.output)
    return

  def no_configure(self):
    if self.staticLibraries:
      self.setStaticLinker()
    return
