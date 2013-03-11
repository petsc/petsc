from __future__ import generators
import config.base

import os

# not sure how to handle this with 'self' so its outside the class
def noCheck(command, status, output, error):
  return

try:
  any
except NameError:
  def any(lst):
    return reduce(lambda x,y:x or y,lst,False)

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.usedMPICompilers = 0
    self.mainLanguage = 'C'
    return

  def __str__(self):
    desc = ['Compilers:']
    if hasattr(self, 'CC'):
      self.pushLanguage('C')
      desc.append('  C Compiler:         '+self.getCompiler()+' '+self.getCompilerFlags())
      if not self.getLinker() == self.getCompiler(): desc.append('  C Linker:           '+self.getLinker()+' '+self.getLinkerFlags())
      self.popLanguage()
    if hasattr(self, 'CUDAC'):
      self.pushLanguage('CUDA')
      desc.append('  CUDA Compiler:      '+self.getCompiler()+' '+self.getCompilerFlags())
      if not self.getLinker() == self.getCompiler(): desc.append('  CUDA Linker:        '+self.getLinker()+' '+self.getLinkerFlags())
      self.popLanguage()
    if hasattr(self, 'CXX'):
      self.pushLanguage('Cxx')
      desc.append('  C++ Compiler:       '+self.getCompiler()+' '+self.getCompilerFlags())
      if not self.getLinker() == self.getCompiler(): desc.append('  C++ Linker:         '+self.getLinker()+' '+self.getLinkerFlags())
      self.popLanguage()
    if hasattr(self, 'FC'):
      self.pushLanguage('FC')
      desc.append('  Fortran Compiler:   '+self.getCompiler()+' '+self.getCompilerFlags())
      if not self.getLinker() == self.getCompiler(): desc.append('  Fortran Linker:     '+self.getLinker()+' '+self.getLinkerFlags())
      self.popLanguage()
    desc.append('Linkers:')
    if hasattr(self, 'staticLinker'):
      desc.append('  Static linker:   '+self.getSharedLinker()+' '+self.AR_FLAGS)
    elif hasattr(self, 'sharedLinker'):
      desc.append('  Shared linker:   '+self.getSharedLinker()+' '+self.getSharedLinkerFlags())
    if hasattr(self, 'dynamicLinker'):
      desc.append('  Dynamic linker:   '+self.getDynamicLinker()+' '+self.getDynamicLinkerFlags())
    return '\n'.join(desc)+'\n'

  def setupHelp(self, help):
    import nargs

    help.addArgument('Compilers', '-with-cpp=<prog>', nargs.Arg(None, None, 'Specify the C preprocessor'))
    help.addArgument('Compilers', '-CPP=<prog>',            nargs.Arg(None, None, 'Specify the C preprocessor'))
    help.addArgument('Compilers', '-CPPFLAGS=<string>',     nargs.Arg(None, None, 'Specify the C preprocessor options'))
    help.addArgument('Compilers', '-with-cc=<prog>',  nargs.Arg(None, None, 'Specify the C compiler'))
    help.addArgument('Compilers', '-CC=<prog>',             nargs.Arg(None, None, 'Specify the C compiler'))
    help.addArgument('Compilers', '-CFLAGS=<string>',       nargs.Arg(None, None, 'Specify the C compiler options'))
    help.addArgument('Compilers', '-CC_LINKER_FLAGS=<string>',        nargs.Arg(None, [], 'Specify the C linker flags'))

    help.addArgument('Compilers', '-CXXPP=<prog>',          nargs.Arg(None, None, 'Specify the C++ preprocessor'))
    help.addArgument('Compilers', '-CXXCPPFLAGS=<string>',  nargs.Arg(None, None, 'Specify the C++ preprocessor options'))
    help.addArgument('Compilers', '-with-cxx=<prog>', nargs.Arg(None, None, 'Specify the C++ compiler'))
    help.addArgument('Compilers', '-CXX=<prog>',            nargs.Arg(None, None, 'Specify the C++ compiler'))
    help.addArgument('Compilers', '-CXXFLAGS=<string>',     nargs.Arg(None, None, 'Specify the C++ compiler options'))
    help.addArgument('Compilers', '-CXX_CXXFLAGS=<string>', nargs.Arg(None, '',   'Specify the C++ compiler-only options'))
    help.addArgument('Compilers', '-CXX_LINKER_FLAGS=<string>',       nargs.Arg(None, [], 'Specify the C++ linker flags'))

    help.addArgument('Compilers', '-with-fc=<prog>',  nargs.Arg(None, None, 'Specify the Fortran compiler'))
    help.addArgument('Compilers', '-FC=<prog>',             nargs.Arg(None, None, 'Specify the Fortran compiler'))
    help.addArgument('Compilers', '-FFLAGS=<string>',       nargs.Arg(None, None, 'Specify the Fortran compiler options'))
    help.addArgument('Compilers', '-FC_LINKER_FLAGS=<string>',        nargs.Arg(None, [], 'Specify the FC linker flags'))

    help.addArgument('Compilers', '-with-gnu-compilers=<bool>',      nargs.ArgBool(None, 1, 'Try to use GNU compilers'))
    help.addArgument('Compilers', '-with-vendor-compilers=<vendor as string>', nargs.Arg(None, '', 'Try to use vendor compilers (no argument all vendors, 0 no vendors)'))

    help.addArgument('Compilers', '-with-large-file-io=<bool>', nargs.ArgBool(None, 0, 'Allow IO with files greater then 2 GB'))

    help.addArgument('Compilers', '-CUDAPP=<prog>',        nargs.Arg(None, None, 'Specify the CUDA preprocessor'))
    help.addArgument('Compilers', '-CUDAPPFLAGS=<string>', nargs.Arg(None, None, 'Specify the CUDA preprocessor options'))
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
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.headers = framework.require('config.headers', None)
    self.libraries = framework.require('config.libraries', None)
    return

  def isNAG(compiler):
    '''Returns true if the compiler is a NAG F90 compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' -V',checkCommand = noCheck)
      output = output + error
      if output.find('NAGWare Fortran') >= 0 or output.find('The Numerical Algorithms Group Ltd') >= 0:
        return 1
    except RuntimeError:
      pass
    return 0
  isNAG = staticmethod(isNAG)

  def isGNU(compiler):
    '''Returns true if the compiler is a GNU compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help')
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
    return 0
  isGNU = staticmethod(isGNU)

  def isClang(compiler):
    '''Returns true if the compiler is a Clang/LLVM compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help')
      output = output + error
      return any([s in output for s in ['Emit Clang AST']])
    except RuntimeError:
      pass
    return 0
  isClang = staticmethod(isClang)

  def isGfortran45x(compiler):
    '''returns true if the compiler is gfortran-4.5.x'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version')
      output = output +  error
      import re
      if re.match(r'GNU Fortran \(.*\) (4.5.\d+|4.6.0 20100703)', output):
        return 1
    except RuntimeError:
      pass
    return 0
  isGfortran45x = staticmethod(isGfortran45x)

  def isGfortran46plus(compiler):
    '''returns true if the compiler is gfortran-4.6.x or later'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version')
      output = output +  error
      import re
      if re.match(r'GNU Fortran \(.*\) (4.([6789]|\d{2,}).\d+)', output):
        return 1
    except RuntimeError:
      pass
    return 0
  isGfortran46plus = staticmethod(isGfortran46plus)


  def isG95(compiler):
    '''Returns true if the compiler is g95'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help')
      output = output + error
      if output.find('Unrecognised option --help passed to ld') >=0:    # NAG f95 compiler
        return 0
      if output.find('http://www.g95.org') >= 0:
        return 1
    except RuntimeError:
      pass
    return 0
  isG95 = staticmethod(isG95)

  def isCompaqF90(compiler):
    '''Returns true if the compiler is Compaq f90'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help')
      output = output + error
      if output.find('Unrecognised option --help passed to ld') >=0:    # NAG f95 compiler
        return 0
      if output.find('Compaq Visual Fortran') >= 0 or output.find('Digital Visual Fortran') >=0 :
        return 1
    except RuntimeError:
      pass
    return 0
  isCompaqF90 = staticmethod(isCompaqF90)

  def isSun(compiler):
    '''Returns true if the compiler is a Sun compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' -flags')
      output = output + error
      if output.find('Unrecognised option --help passed to ld') >=0:    # NAG f95 compiler
        return 0
      if output.find('http://www.sun.com') >= 0 or output.find('http://docs.sun.com') >=0:
        return 1
    except RuntimeError:
      pass
    return 0
  isSun = staticmethod(isSun)

  def isIBM(compiler):
    '''Returns true if the compiler is a IBM compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' -flags')
      output = output + error
      #
      # Do not know what to look for for IBM compilers
      #
      return 0
    except RuntimeError:
      pass
    return 0
  isIBM = staticmethod(isIBM)

  def isIntel(compiler):
    '''Returns true if the compiler is a Intel compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --help')
      output = output + error
      if output.find('Intel Corporation') >= 0 :
        return 1
    except RuntimeError:
      pass
    return 0
  isIntel = staticmethod(isIntel)

  def isCray(compiler):
    '''Returns true if the compiler is a Cray compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' -V')
      output = output + error
      if output.find('Cray Standard C') >= 0 or output.find('Cray C++') >= 0 or output.find('Cray Fortran') >= 0:
        return 1
    except RuntimeError:
      pass
    return 0
  isCray = staticmethod(isCray)
  
  def isCrayVector(compiler):
    '''Returns true if the compiler is a Cray compiler for a Cray Vector system'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' -VV')
      output = output + error
      if not status and output.find('x86') >= 0:
        return 0
      elif not status:
        return 1
      else:
        return 0
    except RuntimeError:
      pass
    return 0
  isCrayVector = staticmethod(isCrayVector)
  

  def isPGI(compiler):
    '''Returns true if the compiler is a PGI compiler'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' -V',checkCommand = noCheck)
      output = output + error
      if output.find('The Portland Group') >= 0:
        return 1
    except RuntimeError:
      pass
    return 0
  isPGI = staticmethod(isPGI)

  def isSolarisAR(ar):
    '''Returns true AR is solaris'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(ar + ' -V',checkCommand = noCheck)
      output = output + error
      if output.find('Software Generation Utilities') >= 0:
        return 1
    except RuntimeError:
      pass
    return 0
  isSolarisAR = staticmethod(isSolarisAR)

  def isAIXAR(ar):
    '''Returns true AR is AIX'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(ar + ' -V',checkCommand = noCheck)
      output = output + error
      if output.find('[-X{32|64|32_64|d64|any}]') >= 0:
        return 1
    except RuntimeError:
      pass
    return 0
  isAIXAR = staticmethod(isAIXAR)

  
  def isLinux():
    '''Returns true if system is linux'''
    (output, error, status) = config.base.Configure.executeShellCommand('uname -s')
    if not status and output.lower().strip().find('linux') >= 0:
      return 1
    else:
      return 0
  isLinux = staticmethod(isLinux)

  def isCygwin():
    '''Returns true if system is cygwin'''
    (output, error, status) = config.base.Configure.executeShellCommand('uname -s')
    if not status and output.lower().strip().find('cygwin') >= 0:
      return 1
    else:
      return 0
  isCygwin = staticmethod(isCygwin)

  def isSolaris():
    '''Returns true if system is solaris'''
    (output, error, status) = config.base.Configure.executeShellCommand('uname -s')
    if not status and output.lower().strip().find('sunos') >= 0:
      return 1
    else:
      return 0
  isSolaris = staticmethod(isSolaris)

  def isDarwin():
    '''Returns true if system is Darwin/MacOSX'''
    (output, error, status) = config.base.Configure.executeShellCommand('uname -s')
    if not status:
      return output.lower().strip() == 'darwin'
    return 0
  isDarwin = staticmethod(isDarwin)

  def isWindows(compiler):
    '''Returns true if the compiler is a Windows compiler'''
    if compiler in ['icl', 'cl', 'bcc32', 'ifl', 'df']:
      return 1
    if compiler in ['ifort','f90'] and Configure.isCygwin():
      return 1
    if compiler in ['lib', 'tlib']:
      return 1
    return 0
  isWindows = staticmethod(isWindows)

  def useMPICompilers(self):
    if ('with-cc' in self.argDB and self.argDB['with-cc'] != '0') or 'CC' in self.argDB:
      return 0
    if ('with-cxx' in self.argDB and self.argDB['with-cxx'] != '0') or 'CXX' in self.argDB:
      return 0
    if ('with-fc' in self.argDB and self.argDB['with-fc'] != '0') or 'FC' in self.argDB:
      return 0
    if 'with-mpi' in self.argDB and self.argDB['with-mpi'] and self.argDB['with-mpi-compilers'] and not self.argDB['download-mpich'] == 1 and not self.argDB['download-openmpi'] == 1:
      return 1
    return 0

  def checkVendor(self):
    '''Determine the compiler vendor'''
    self.vendor = self.framework.argDB['with-vendor-compilers']
    if self.framework.argDB['with-vendor-compilers'] == 'no' or self.framework.argDB['with-vendor-compilers'] == 'false':
      self.vendor = None
    if self.framework.argDB['with-vendor-compilers'] == '1' or self.framework.argDB['with-vendor-compilers'] == 'yes' or self.framework.argDB['with-vendor-compilers'] == 'true':
      self.vendor = ''
    self.logPrint('Compiler vendor is "'+str(self.vendor)+'"')
    return

  def checkInitialFlags(self):
    '''Initialize the compiler and linker flags'''
    for language in ['C', 'CUDA', 'Cxx', 'FC']:
      self.pushLanguage(language)
      for flagsArg in [self.getCompilerFlagsName(language), self.getCompilerFlagsName(language, 1), self.getLinkerFlagsName(language)]:
        if flagsArg in self.argDB: setattr(self, flagsArg, self.argDB[flagsArg])
        else: setattr(self, flagsArg, '')
        self.framework.logPrint('Initialized '+flagsArg+' to '+str(getattr(self, flagsArg)))
      self.popLanguage()
    for flagsArg in ['CPPFLAGS', 'CUDAPPFLAGS', 'CXXCPPFLAGS', 'CC_LINKER_FLAGS', 'CXX_LINKER_FLAGS', 'FC_LINKER_FLAGS', 'CUDAC_LINKER_FLAGS','sharedLibraryFlags', 'dynamicLibraryFlags']:
      if flagsArg in self.argDB: setattr(self, flagsArg, self.argDB[flagsArg])
      else: setattr(self, flagsArg, '')
      self.framework.logPrint('Initialized '+flagsArg+' to '+str(getattr(self, flagsArg)))
    if 'LIBS' in self.argDB:
      self.LIBS = self.argDB['LIBS']
    else:
      self.LIBS = ''
    return

  def checkCompiler(self, language):
    '''Check that the given compiler is functional, and if not raise an exception'''
    self.pushLanguage(language)
    if not self.checkCompile():
      msg = 'Cannot compile '+language+' with '+self.getCompiler()+'.'
      self.popLanguage()
      raise RuntimeError(msg)
    if language == 'CUDA': # do not check CUDA linker since it is never used (and is broken on Mac with -m64)
      self.popLanguage()
      return
    if not self.checkLink():
      msg = 'Cannot compile/link '+language+' with '+self.getCompiler()+'.'
      self.popLanguage()
      raise RuntimeError(msg)
    if not self.framework.argDB['with-batch']:
      if not self.checkRun():
        msg = 'Cannot run executables created with '+language+'. If this machine uses a batch system \nto submit jobs you will need to configure using ./configure with the additional option  --with-batch.\n Otherwise there is problem with the compilers. Can you compile and run code with your C/C++ (and maybe Fortran) compilers?\n'
        if self.isIntel(self.getCompiler()):
          msg = msg + 'See http://www.mcs.anl.gov/petsc/documentation/faq.html#libimf'
        self.popLanguage()
        raise OSError(msg)
    self.popLanguage()
    return

  def generateCCompilerGuesses(self):
    '''Determine the C compiler using CC, then --with-cc, then MPI, then GNU, then vendors
       - Any given category can be excluded'''
    if hasattr(self, 'CC'):
      yield self.CC
    elif self.framework.argDB.has_key('with-cc'):
      if self.isWindows(self.framework.argDB['with-cc']):
        yield 'win32fe '+self.framework.argDB['with-cc']
      else:
        yield self.framework.argDB['with-cc']
      raise RuntimeError('C compiler you provided with -with-cc='+self.framework.argDB['with-cc']+' does not work')
    elif self.framework.argDB.has_key('CC'):
      if self.isWindows(self.framework.argDB['CC']):
        yield 'win32fe '+self.framework.argDB['CC']
      else:
        yield self.framework.argDB['CC']
      raise RuntimeError('C compiler you provided with -CC='+self.framework.argDB['CC']+' does not work')
    elif self.useMPICompilers() and 'with-mpi-dir' in self.argDB and os.path.isdir(os.path.join(self.argDB['with-mpi-dir'], 'bin')):
      self.usedMPICompilers = 1
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpicc')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpcc')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'hcc')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpcc_r')
      self.usedMPICompilers = 0
      raise RuntimeError('MPI compiler wrappers in '+self.framework.argDB['with-mpi-dir']+'/bin do not work. See http://www.mcs.anl.gov/petsc/documentation/faq.html#mpi-compilers')
    else:
      if self.useMPICompilers():
        self.usedMPICompilers = 1
        if Configure.isGNU('mpicc') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpicc'
        if Configure.isGNU('hcc') and self.framework.argDB['with-gnu-compilers']:
          yield 'hcc'
        if not Configure.isGNU('mpicc') and (not self.vendor is None):
          yield 'mpicc'
        if not Configure.isGNU('hcc') and (not self.vendor is None):
          yield 'hcc'
        if not self.vendor is None:
          yield 'mpcc_r'
          yield 'mpcc'
          yield 'mpxlc'
        self.usedMPICompilers = 0
      vendor = self.vendor
      if (not vendor) and self.framework.argDB['with-gnu-compilers']:
        yield 'gcc'
        if Configure.isGNU('cc'):
          yield 'cc'     
      if not self.vendor is None:
        if not vendor and not Configure.isGNU('cc'):
          yield 'cc'
        if vendor == 'borland' or not vendor:
          yield 'win32fe bcc32'
        if vendor == 'kai' or not vendor:
          yield 'kcc'
        if vendor == 'ibm' or not vendor:
          yield 'xlc'
        if vendor == 'intel' or not vendor:
          yield 'icc'
          yield 'ecc'          
          yield 'win32fe icl'
        if vendor == 'microsoft' or not vendor:
          yield 'win32fe cl'
        if vendor == 'portland' or not vendor:
          yield 'pgcc'
        if vendor == 'solaris' or not vendor:
          if not Configure.isGNU('cc'):
            yield 'cc'
      # duplicate code
      if self.framework.argDB['with-gnu-compilers']:
        yield 'gcc'
        if Configure.isGNU('cc'):
          yield 'cc'     
    return

  def checkCCompiler(self):
    '''Locate a functional C compiler'''
    if 'with-cc' in self.framework.argDB and self.framework.argDB['with-cc'] == '0':
      raise RuntimeError('A functional C compiler is necessary for configure, cannot use --with-cc=0')
    for compiler in self.generateCCompilerGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'CC'):
          self.checkCompiler('C')
          break
      except RuntimeError, e:
        import os

        import sys,traceback
        traceback.print_tb(sys.exc_info()[2])
        self.logPrint('Error testing C compiler: '+str(e))
        if os.path.basename(self.CC) == 'mpicc':
          self.framework.logPrint(' MPI installation '+str(self.CC)+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI.')
        self.delMakeMacro('CC')
        del self.CC
    if not hasattr(self, 'CC'):
      raise RuntimeError('Could not locate a functional C compiler')
    return

  def generateCPreprocessorGuesses(self):
    '''Determines the C preprocessor from CPP, then --with-cpp, then the C compiler'''
    if 'with-cpp' in self.framework.argDB:
      yield self.framework.argDB['with-cpp']
    elif 'CPP' in self.framework.argDB:
      yield self.framework.argDB['CPP']
    else:
      yield self.CC+' -E'
      yield self.CC+' --use cpp32'
      yield 'gcc -E'
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
      except RuntimeError, e:
        self.popLanguage()
    raise RuntimeError('Cannot find a C preprocessor')
    return

  def generateCUDACompilerGuesses(self):
    '''Determine the CUDA compiler using CUDAC, then --with-cudac, then vendors
       - Any given category can be excluded'''
    if hasattr(self, 'CUDAC'):
      yield self.CUDAC
    elif self.framework.argDB.has_key('with-cudac'):
      yield self.framework.argDB['with-cudac']
      raise RuntimeError('CUDA compiler you provided with -with-cudac='+self.framework.argDB['with-cudac']+' does not work')
    elif self.framework.argDB.has_key('CUDAC'):
      yield self.framework.argDB['CUDAC']
      raise RuntimeError('CUDA compiler you provided with -CUDAC='+self.framework.argDB['CUDAC']+' does not work')
    else:
      vendor = self.vendor
      if not self.vendor is None:
        if vendor == 'nvidia' or not vendor:
          yield 'nvcc'
      yield 'nvcc'     
    return

  def checkCUDACompiler(self):
    '''Locate a functional CUDA compiler'''
    if 'with-cudac' in self.framework.argDB and self.framework.argDB['with-cudac'] == '0':
      if 'CUDAC' in self.framework.argDB:
        del self.framework.argDB['CUDAC']
      return
    for compiler in self.generateCUDACompilerGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'CUDAC'):
          self.checkCompiler('CUDA')
          # Put version info into the log
          compilerVersion = self.executeShellCommand(self.CUDAC+' --version')
          compilerVersion = compilerVersion[0]
          compilerVersion = compilerVersion.split()
          i = 0
          for word in compilerVersion:
            i = i+1
            if word == 'release':
              break
          self.compilerVersionCUDA = compilerVersion[i].strip(',')
          break
      except RuntimeError, e:
        self.logPrint('Error testing CUDA compiler: '+str(e))
        self.delMakeMacro('CUDAC')
        del self.CUDAC
    return

  def generateCUDAPreprocessorGuesses(self):
    '''Determines the CUDA preprocessor from --with-cudacpp, then CUDAPP, then the CUDA compiler'''
    if 'with-cudacpp' in self.framework.argDB:
      yield self.framework.argDB['with-cudacpp']
    elif 'CUDAPP' in self.framework.argDB:
      yield self.framework.argDB['CUDAPP']
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
      except RuntimeError, e:
        self.popLanguage()
    return

  def generateCxxCompilerGuesses(self):
    '''Determine the Cxx compiler using CXX, then --with-cxx, then MPI, then GNU, then vendors
       - Any given category can be excluded'''
    import os

    if hasattr(self, 'CXX'):
      yield self.CXX
    elif self.framework.argDB.has_key('with-c++'):
      raise RuntimeError('Keyword --with-c++ is WRONG, use --with-cxx')
    if self.framework.argDB.has_key('with-CC'):
      raise RuntimeError('Keyword --with-CC is WRONG, use --with-cxx')
    
    if self.framework.argDB.has_key('with-cxx'):
      if self.framework.argDB['with-cxx'] == 'gcc': raise RuntimeError('Cannot use C compiler gcc as the C++ compiler passed in with --with-cxx')
      if self.isWindows(self.framework.argDB['with-cxx']):
        yield 'win32fe '+self.framework.argDB['with-cxx']
      else:
        yield self.framework.argDB['with-cxx']
      raise RuntimeError('C++ compiler you provided with -with-cxx='+self.framework.argDB['with-cxx']+' does not work')
    elif self.framework.argDB.has_key('CXX'):
      if self.isWindows(self.framework.argDB['CXX']):
        yield 'win32fe '+self.framework.argDB['CXX']
      else:
        yield self.framework.argDB['CXX']
      raise RuntimeError('C++ compiler you provided with -CXX='+self.framework.argDB['CXX']+' does not work')
    elif self.useMPICompilers() and 'with-mpi-dir' in self.argDB and os.path.isdir(os.path.join(self.argDB['with-mpi-dir'], 'bin')):
      self.usedMPICompilers = 1
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpicxx')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'hcp')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpic++')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpiCC')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpCC_r')
      self.usedMPICompilers = 0
      raise RuntimeError('bin/<mpiCC,mpicxx,hcp,mpCC_r> you provided with -with-mpi-dir='+self.framework.argDB['with-mpi-dir']+' does not work')
    else:
      if self.useMPICompilers():
        self.usedMPICompilers = 1
        if Configure.isGNU('mpicxx') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpicxx'
        if not Configure.isGNU('mpicxx') and (not self.vendor is None):
          yield 'mpicxx'
        if Configure.isGNU('mpiCC') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpiCC'
        if not Configure.isGNU('mpiCC') and (not self.vendor is None):
          yield 'mpiCC'
        if Configure.isGNU('mpic++') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpic++'
        if not Configure.isGNU('mpic++') and (not self.vendor is None):
          yield 'mpic++'
        if not self.vendor is None:
          yield 'mpCC_r'
          yield 'mpCC'          
        self.usedMPICompilers = 0
      vendor = self.vendor
      if (not vendor) and self.framework.argDB['with-gnu-compilers']:
        yield 'g++'
        if Configure.isGNU('c++'):
          yield 'c++'
      if not self.vendor is None:
        if not vendor:
          if not Configure.isGNU('c++'):
            yield 'c++'
          if not Configure.isGNU('CC'):
            yield 'CC'
          yield 'cxx'
          yield 'cc++'
        if vendor == 'borland' or not vendor:
          yield 'win32fe bcc32'
        if vendor == 'ibm' or not vendor:
          yield 'xlC'
        if vendor == 'intel' or not vendor:
          yield 'icpc'
          yield 'ccpc'          
          yield 'icc'
          yield 'ecc'          
          yield 'win32fe icl'
        if vendor == 'microsoft' or not vendor:
          yield 'win32fe cl'
        if vendor == 'portland' or not vendor:
          yield 'pgCC'
        if vendor == 'solaris':
          yield 'CC'
      #duplicate code
      if self.framework.argDB['with-gnu-compilers']:
        yield 'g++'
        if Configure.isGNU('c++'):
          yield 'c++'
    return

  def checkCxxCompiler(self):
    '''Locate a functional Cxx compiler'''
    if 'with-cxx' in self.framework.argDB and self.framework.argDB['with-cxx'] == '0':
      if 'CXX' in self.framework.argDB:
        del self.framework.argDB['CXX']
      return
    for compiler in self.generateCxxCompilerGuesses():
      # Determine an acceptable extensions for the C++ compiler
      for ext in ['.cc', '.cpp', '.C']:
        self.framework.getCompilerObject('Cxx').sourceExtension = ext
        try:
          if self.getExecutable(compiler, resultName = 'CXX'):
            self.checkCompiler('Cxx')
            break
        except RuntimeError, e:
          import os

          self.logPrint('Error testing C++ compiler: '+str(e))
          if os.path.basename(self.CXX) in ['mpicxx', 'mpiCC']:
            self.logPrint('  MPI installation '+str(self.CXX)+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI.')
          self.delMakeMacro('CXX')
          del self.CXX
      if hasattr(self, 'CXX'):
        break
    return

  def generateCxxPreprocessorGuesses(self):
    '''Determines the Cxx preprocessor from CXXCPP, then --with-cxxcpp, then the Cxx compiler'''
    if 'with-cxxcpp' in self.framework.argDB:
      yield self.framework.argDB['with-cxxcpp']
    elif 'CXXCPP' in self.framework.argDB:
      yield self.framework.argDB['CXXCPP']
    else:
      yield self.CXX+' -E'
      yield self.CXX+' --use cpp32'
      yield 'g++ -E'
    return

  def checkCxxPreprocessor(self):
    '''Locate a functional Cxx preprocessor'''
    if not hasattr(self, 'CXX'):
      return
    for compiler in self.generateCxxPreprocessorGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'CXXCPP'):
          self.pushLanguage('Cxx')
          if not self.checkPreprocess('#include <cstdlib>\n'):
            raise RuntimeError('Cannot preprocess Cxx with '+self.CXXCPP+'.')
          self.popLanguage()
          break
      except RuntimeError, e:
        import os

        if os.path.basename(self.CXXCPP) in ['mpicxx', 'mpiCC']:
          self.framework.logPrint('MPI installation '+self.getCompiler()+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI')
        self.popLanguage()
        self.delMakeMacro('CCCPP')
        del self.CXXCPP
    return

  def generateFortranCompilerGuesses(self):
    '''Determine the Fortran compiler using FC, then --with-fc, then MPI, then GNU, then vendors
       - Any given category can be excluded'''
    import os

    if hasattr(self, 'FC'):
      yield self.FC
    elif self.framework.argDB.has_key('with-fc'):
      if self.isWindows(self.framework.argDB['with-fc']):
        yield 'win32fe '+self.framework.argDB['with-fc']
      else:
        yield self.framework.argDB['with-fc']
      raise RuntimeError('Fortran compiler you provided with --with-fc='+self.framework.argDB['with-fc']+' does not work')
    elif self.framework.argDB.has_key('FC'):
      if self.isWindows(self.framework.argDB['FC']):
        yield 'win32fe '+self.framework.argDB['FC']
      else:
        yield self.framework.argDB['FC']
      yield self.framework.argDB['FC']
      raise RuntimeError('Fortran compiler you provided with -FC='+self.framework.argDB['FC']+' does not work')
    elif self.useMPICompilers() and 'with-mpi-dir' in self.argDB and os.path.isdir(os.path.join(self.argDB['with-mpi-dir'], 'bin')):
      self.usedMPICompilers = 1
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpif90')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpif77')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpxlf95_r')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpxlf90_r')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpxlf_r')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpf90')
      yield os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpf77')
      self.usedMPICompilers = 0
      if os.path.isfile(os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpif90')) or os.path.isfile((os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', 'mpif77'))):
        raise RuntimeError('bin/mpif90[f77] you provided with --with-mpi-dir='+self.framework.argDB['with-mpi-dir']+' does not work\nRun with --with-fc=0 if you wish to use this MPI and disable Fortran')
    else:
      if self.useMPICompilers():
        self.usedMPICompilers = 1
        if Configure.isGNU('mpif90') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpif90'
        if not Configure.isGNU('mpif90') and (not self.vendor is None):
          yield 'mpif90'
        if Configure.isGNU('mpif77') and self.framework.argDB['with-gnu-compilers']:
          yield 'mpif77'
        if not Configure.isGNU('mpif77') and (not self.vendor is None):
          yield 'mpif77'
        if not self.vendor is None:
          yield 'mpxlf_r'
          yield 'mpxlf'          
          yield 'mpf90'
          yield 'mpf77'
        self.usedMPICompilers = 0
      vendor = self.vendor
      if (not vendor) and self.framework.argDB['with-gnu-compilers']:
        yield 'gfortran'
        yield 'g95'
        yield 'g77'
        if Configure.isGNU('f77'):
          yield 'f77'
      if not self.vendor is None:
        if vendor == 'ibm' or not vendor:
          yield 'xlf'
          yield 'xlf90'
        if not vendor or vendor in ['absoft', 'cray', 'dec', 'hp', 'sgi']:
          yield 'f90'
        if vendor == 'lahaye' or not vendor:
          yield 'lf95'
        if vendor == 'intel' or not vendor:
          yield 'win32fe ifort'
          yield 'win32fe ifl'
          yield 'ifort'
          yield 'ifc'
          yield 'efc'          
        if vendor == 'portland' or not vendor:
          yield 'pgf90'
          yield 'pgf77'
        if vendor == 'solaris' or not vendor:
          yield 'f95'
          yield 'f90'
          if not Configure.isGNU('f77'):
            yield 'f77'
      #duplicate code
      if self.framework.argDB['with-gnu-compilers']:
        yield 'gfortran'
        yield 'g95'
        yield 'g77'
        if Configure.isGNU('f77'):
          yield 'f77'
    return

  def checkFortranCompiler(self):
    '''Locate a functional Fortran compiler'''
    if 'with-fc' in self.framework.argDB and self.framework.argDB['with-fc'] == '0':
      if 'FC' in self.framework.argDB:
        del self.framework.argDB['FC']
      return
    for compiler in self.generateFortranCompilerGuesses():
      try:
        if self.getExecutable(compiler, resultName = 'FC'):
          self.checkCompiler('FC')
          break
      except RuntimeError, e:
        self.logPrint('Error testing Fortran compiler: '+str(e))
        if os.path.basename(self.FC) in ['mpif90', 'mpif77']:
          self.framework.logPrint(' MPI installation '+str(self.FC)+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI.')
        self.delMakeMacro('FC')
        del self.FC
    return

  def checkFortranComments(self):
    '''Make sure fortran comment "!" works'''
    self.pushLanguage('FC')
    if not self.checkCompile('! comment'):
      raise RuntimeError(self.getCompiler()+' cannot process fortran comments.')
    self.framework.logPrint('Fortran comments can use ! in column 1')
    self.popLanguage()
    return

  def containsInvalidFlag(self, output):
    '''If the output contains evidence that an invalid flag was used, return True'''
    if (output.find('Unrecognized command line option') >= 0 or output.find('Unrecognised command line option') >= 0 or
        output.find('unrecognized command line option') >= 0 or output.find('unrecognized option') >= 0 or output.find('unrecognised option') >= 0 or
        output.find('not recognized') >= 0 or output.find('not recognised') >= 0 or
        output.find('unknown option') >= 0 or output.find('unknown flag') >= 0 or output.find('Unknown switch') >= 0 or
        output.find('ignoring option') >= 0 or output.find('ignored') >= 0 or
        output.find('argument unused') >= 0 or
        # When checking for the existence of 'attribute'
        output.find('is unsupported and will be skipped') >= 0 or
        output.find('illegal option') >= 0 or output.find('Invalid option') >= 0 or
        (output.find('bad ') >= 0 and output.find(' option') >= 0) or
        output.find('linker input file unused because linking not done') >= 0 or
        output.find('PETSc Error') >= 0 or
        output.find('Unbekannte Option') >= 0 or
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
    # Please comment each entry and provide an example line
    if status:
      valid = 0
      self.framework.logPrint('Rejecting compiler flag '+flag+' due to nonzero status from link')
    # Lahaye F95
    if output.find('Invalid suboption') >= 0:
      valid = 0
    if self.containsInvalidFlag(output):
      valid = 0
      self.framework.logPrint('Rejecting compiler flag '+flag+' due to \n'+output)
    setattr(self, flagsArg, oldFlags)
    return valid

  def insertCompilerFlag(self, flag, compilerOnly):
    '''DANGEROUS: Put in the compiler flag without checking'''
    flagsArg = self.getCompilerFlagsArg(compilerOnly)
    setattr(self, flagsArg, getattr(self, flagsArg)+' '+flag)
    self.framework.log.write('Added '+self.language[-1]+' compiler flag '+flag+'\n')
    return

  def addCompilerFlag(self, flag, includes = '', body = '', extraflags = '', compilerOnly = 0):
    '''Determine whether the compiler accepts the given flag, and add it if valid, otherwise throw an exception'''
    if self.checkCompilerFlag(flag+' '+extraflags, includes, body, compilerOnly):
      self.insertCompilerFlag(flag, compilerOnly)
      return
    raise RuntimeError('Bad compiler flag: '+flag)

  def checkPIC(self):
    '''Determine the PIC option for each compiler
       - There needs to be a test that checks that the functionality is actually working'''
    self.usePIC = 0
    useSharedLibraries = 'with-shared-libraries' in self.framework.argDB and self.framework.argDB['with-shared-libraries']
    useDynamicLoading  = 'with-dynamic-loading'  in self.framework.argDB and self.framework.argDB['with-dynamic-loading']
    if not self.framework.argDB['with-pic'] and not useSharedLibraries and not useDynamicLoading:
      self.framework.logPrint("Skip checking PIC options on user request")
      return
    languages = ['C']
    if hasattr(self, 'CXX'):
      languages.append('Cxx')
    if hasattr(self, 'FC'):
      languages.append('FC')
    for language in languages:
      self.pushLanguage(language)
      for testFlag in ['-PIC', '-fPIC', '-KPIC','-qpic']:
        try:
          self.framework.logPrint('Trying '+language+' compiler flag '+testFlag)
          if not self.checkLinkerFlag(testFlag):
            self.framework.logPrint('Rejected '+language+' compiler flag '+testFlag+' because linker cannot handle it')
            continue
          self.framework.logPrint('Adding '+language+' compiler flag '+testFlag)
          self.addCompilerFlag(testFlag, compilerOnly = 1)
          self.isPIC = 1
          break
        except RuntimeError:
          self.framework.logPrint('Rejected '+language+' compiler flag '+testFlag)
      self.popLanguage()
    return

  def checkLargeFileIO(self):
    # check for large file support with 64bit offset
    if not self.framework.argDB['with-large-file-io']:
      return
    languages = ['C']
    if hasattr(self, 'CXX'):
      languages.append('Cxx')
    for language in languages:
      self.pushLanguage(language)
      if self.checkCompile('#include <unistd.h>','#ifndef _LFS64_LARGEFILE \n#error no largefile defines \n#endif'):
        try:
          self.addCompilerFlag('-D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64',compilerOnly=1)
        except RuntimeError, e:
          self.logPrint('Error adding ' +language+ ' flags -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64')
      else:
        self.logPrint('Rejected ' +language+ ' flags -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64')
      self.popLanguage()
    return

  def getArchiverFlags(self, archiver):
    prog = os.path.basename(archiver).split(' ')[0]
    flag = ''
    if 'AR_FLAGS' in self.framework.argDB: 
      flag = self.framework.argDB['AR_FLAGS']
    elif prog.endswith('ar'):
      flag = 'cr'
    elif prog == 'win32fe':
      args = os.path.basename(archiver).split(' ')
      if 'lib' in args:
        flag = '-a'
      elif 'tlib' in args:
        flag = '-a -P512'
    if prog.endswith('ar') and not (self.isSolarisAR(prog) or self.isAIXAR(prog)):
      self.FAST_AR_FLAGS = 'Scq'
    else:
      self.FAST_AR_FLAGS = flag      
    self.framework.addMakeMacro('FAST_AR_FLAGS',self.FAST_AR_FLAGS )
    return flag
  
  def generateArchiverGuesses(self):
    defaultAr = None
    if 'with-ar' in self.framework.argDB:
      if self.isWindows(self.framework.argDB['with-ar']):
        defaultAr = 'win32fe '+self.framework.argDB['with-ar']
      else:
        defaultAr = self.framework.argDB['with-ar']
    envAr = None
    if 'AR' in self.framework.argDB:
      if self.isWindows(self.framework.argDB['AR']):
        envAr = 'win32fe '+self.framework.argDB['AR']
      else:
        envAr = self.framework.argDB['AR']
    defaultRanlib = None
    if 'with-ranlib' in self.framework.argDB:
      defaultRanlib = self.framework.argDB['with-ranlib']
    envRanlib = None
    if 'RANLIB' in self.framework.argDB:
      envRanlib = self.framework.argDB['RANLIB']
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
        self.framework.logPrint('Possible ERROR while running archiver: '+output)
        if status: self.framework.logPrint('ret = '+str(status))
        if error: self.framework.logPrint('error message = {'+error+'}')
        if os.path.isfile(objName):
          os.remove(objName)
        raise RuntimeError('Archiver is not functional')
      return
    def checkRanlib(command, status, output, error):
      if error or status:
        self.framework.logPrint('Possible ERROR while running ranlib: '+output)
        if status: self.framework.logPrint('ret = '+str(status))
        if error: self.framework.logPrint('error message = {'+error+'}')
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
            (output, error, status) = config.base.Configure.executeShellCommand(self.AR+' '+arflags+' '+arcUnix+' '+objName, checkCommand = checkArchive, log = self.framework.log)
            (output, error, status) = config.base.Configure.executeShellCommand(self.RANLIB+' '+arcUnix, checkCommand = checkRanlib, log = self.framework.log)
          except RuntimeError, e:
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
    return self.framework.setSharedLinkerObject(language, self.framework.getLanguageModule(language).StaticLinker(self.framework.argDB))

  def generateSharedLinkerGuesses(self):
    useSharedLibraries = 'with-shared-libraries' in self.framework.argDB and self.framework.argDB['with-shared-libraries']
    useDynamicLoading  = 'with-dynamic-loading'  in self.framework.argDB and self.framework.argDB['with-dynamic-loading']
    if not self.framework.argDB['with-pic'] and not useSharedLibraries and not useDynamicLoading:
      self.setStaticLinker()
      self.staticLinker = self.AR
      self.staticLibraries = 1
      self.LDFLAGS = ''
      yield (self.AR, [], self.AR_LIB_SUFFIX)
      raise RuntimeError('Archiver failed static link check')
    if 'with-shared-ld' in self.framework.argDB:
      yield (self.framework.argDB['with-shared-ld'], [], 'so')
    if 'LD_SHARED' in self.framework.argDB:
      yield (self.framework.argDB['LD_SHARED'], [], 'so')
    if Configure.isDarwin():
      if 'with-shared-ld' in self.framework.argDB:
        yield (self.framework.argDB['with-dynamic-ld'], ['-dynamiclib -single_module', '-undefined dynamic_lookup', '-multiply_defined suppress'], 'dylib')
      #yield ('libtool', ['-noprebind','-dynamic','-single_module','-flat_namespace -undefined warning','-multiply_defined suppress'], 'dylib')
      if hasattr(self, 'CXX') and self.mainLanguage == 'Cxx':
#        yield ("g++", ['-dynamiclib -single_module', '-undefined dynamic_lookup', '-multiply_defined suppress'], 'dylib')
        yield (self.CXX, ['-dynamiclib -single_module', '-undefined dynamic_lookup', '-multiply_defined suppress'], 'dylib')        
#      yield ("gcc", ['-dynamiclib -single_module', '-undefined dynamic_lookup', '-multiply_defined suppress'], 'dylib')
      yield (self.CC, ['-dynamiclib -single_module', '-undefined dynamic_lookup', '-multiply_defined suppress'], 'dylib')      
    if hasattr(self, 'CXX') and self.mainLanguage == 'Cxx':
      # C++ compiler default
      yield (self.CXX, ['-shared'], 'so')
      yield (self.CXX, ['-dynamic'], 'so')
    # C compiler default
    yield (self.CC, ['-shared'], 'so')
    yield (self.CC, ['-dynamic'], 'so')
    yield (self.CC, ['-qmkshrobj'], 'so')
    # Solaris default
    if Configure.isSolaris():
      if hasattr(self, 'CXX') and self.mainLanguage == 'Cxx':
        yield (self.CXX, ['-G'], 'so')
      yield (self.CC, ['-G'], 'so')
    # Default to static linker
    self.setStaticLinker()
    self.staticLinker = self.AR
    self.staticLibraries = 1
    self.LDFLAGS = ''
    yield (self.AR, [], self.AR_LIB_SUFFIX)
    raise RuntimeError('Archiver failed static link check')

  def checkSharedLinker(self):
    '''Check that the linker can produce shared libraries'''
    self.sharedLibraries = 0
    self.staticLibraries = 0
    for linker, flags, ext in self.generateSharedLinkerGuesses():
      self.logPrint('Checking shared linker '+linker+' using flags '+str(flags))
      if self.getExecutable(linker, resultName = 'LD_SHARED'):
        flagsArg = self.getLinkerFlagsArg()
        goodFlags = filter(self.checkLinkerFlag, flags)
        testMethod = 'foo'
        self.sharedLinker = self.LD_SHARED
        self.sharedLibraryFlags = goodFlags
        self.sharedLibraryExt = ext
        # using printf appears to correctly identify non-pic code on X86_64
        if self.checkLink(includes = '#include <stdio.h>\nint '+testMethod+'(void) {printf("hello");\nreturn 0;}\n', codeBegin = '', codeEnd = '', cleanup = 0, shared = 1):
          oldLib  = self.linkerObj
          oldLibs = self.LIBS
          self.LIBS += ' -L'+self.tmpDir+' -lconftest'
          if self.checkLink(includes = 'int foo(void);', body = 'int ret = foo();\nif(ret);'):
            os.remove(oldLib)
            self.LIBS = oldLibs
            self.sharedLibraries = 1
            self.logPrint('Using shared linker '+self.sharedLinker+' with flags '+str(self.sharedLibraryFlags)+' and library extension '+self.sharedLibraryExt)
            break
          os.remove(oldLib)
          self.LIBS = oldLibs
        if os.path.isfile(self.linkerObj): os.remove(self.linkerObj)
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
      self.framework.logPrint('Rejecting linker flag '+flag+' due to nonzero status from link')
    if self.containsInvalidFlag(output):
      valid = 0
      self.framework.logPrint('Rejecting '+self.language[-1]+' linker flag '+flag+' due to \n'+output)
    if valid:
      self.framework.logPrint('Valid '+self.language[-1]+' linker flag '+flag)
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
      for testFlag in ['-Wl,-multiply_defined,suppress', '-Wl,-multiply_defined -Wl,suppress', '-Wl,-commons,use_dylibs', '-Wl,-search_paths_first']:
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
      if not Configure.isDarwin():      
        testFlags = ['-Wl,-rpath,', '-R','-rpath ' , '-Wl,-R,']
      else:
        testFlags = []
      # test '-R' before '-Wl,-rpath' for SUN compilers [as cc on linux accepts -Wl,-rpath, but  f90 & CC do not.
      if self.isSun(self.framework.getCompiler()):
        testFlags.insert(0,'-R')
      for testFlag in testFlags:
        self.framework.logPrint('Trying '+language+' linker flag '+testFlag)
        if self.checkLinkerFlag(testFlag+os.path.abspath(os.getcwd())):
          flag = testFlag
          break
        else:
          self.framework.logPrint('Rejected '+language+' linker flag '+testFlag)
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
    if 'with-dynamic-ld' in self.framework.argDB:
      yield (self.framework.argDB['with-dynamic-ld'], [], 'so')
    # Mac OSX
    if Configure.isDarwin():
      if 'with-dynamic-ld' in self.framework.argDB:
        yield (self.framework.argDB['with-dynamic-ld'], ['-dynamiclib -single_module', '-undefined dynamic_lookup', '-multiply_defined suppress'], 'dylib')
      #yield ('libtool', ['-noprebind','-dynamic','-single_module','-flat_namespace -undefined warning','-multiply_defined suppress'], 'dylib')
      if hasattr(self, 'CXX') and self.mainLanguage == 'Cxx':
#        yield ("g++", ['-dynamiclib -single_module', '-undefined dynamic_lookup', '-multiply_defined suppress'], 'dylib')
        yield (self.CXX, ['-dynamiclib -single_module', '-undefined dynamic_lookup', '-multiply_defined suppress'], 'dylib')
#      yield ("gcc", ['-dynamiclib -single_module', '-undefined dynamic_lookup', '-multiply_defined suppress'], 'dylib')
      yield (self.CC, ['-dynamiclib -single_module', '-undefined dynamic_lookup', '-multiply_defined suppress'], 'dylib')
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
    if not self.libraries.add('dl', ['dlopen', 'dlsym', 'dlclose']):
      if not self.libraries.check('', ['dlopen', 'dlsym', 'dlclose']):
        self.logPrint('Dynamic linking disabled since functions dlopen(), dlsym(), and dlclose() were not found')
        return
    for linker, flags, ext in self.generateDynamicLinkerGuesses():
      self.logPrint('Checking dynamic linker '+linker+' using flags '+str(flags))
      if self.getExecutable(linker, resultName = 'dynamicLinker'):
        flagsArg = self.getLinkerFlagsArg()
        goodFlags = filter(self.checkLinkerFlag, flags)
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
          if self.checkLink(includes = '#include<dlfcn.h>', body = code):
            os.remove(oldLib)
            self.dynamicLibraries = 1
            self.logPrint('Using dynamic linker '+self.dynamicLinker+' with flags '+str(self.dynamicLibraryFlags)+' and library extension '+self.dynamicLibraryExt)
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
    if hasattr(self, 'CXXCPP'):
      self.addSubstitution('CXXCPP', self.CXXCPP)
      self.addSubstitution('CXXCPPFLAGS', self.CXXCPPFLAGS)
    if hasattr(self, 'FC'):
      self.addSubstitution('FC', self.FC)
      self.addSubstitution('FFLAGS', self.FFLAGS)
      self.addMakeMacro('FC_LINKER_SLFLAG', self.FCSharedLinkerFlag)
    else:
      self.addSubstitution('FC', '')
    self.addSubstitution('LDFLAGS', self.LDFLAGS)
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

    if 'with-mpi-dir' in self.argDB:
      optcplrs = [(['with-cc','CC'],['mpicc','mpcc','hcc','mpcc_r']),
              (['with-fc','FC'],['mpif90','mpif77','mpxlf95_r','mpxlf90_r','mpxlf_r','mpf90','mpf77']),
              (['with-cxx','CXX'],['mpicxx','hcp','mpic++','mpiCC','mpCC_r'])]
      for opts,cplrs in optcplrs:
        for opt in opts:
          if (opt in self.argDB  and self.argDB[opt] != '0'):
            # check if corresponding mpi wrapper exists
            for cplr in cplrs:
              mpicplr = os.path.join(self.framework.argDB['with-mpi-dir'], 'bin', cplr)
              if os.path.exists(mpicplr):
                msg = '--'+opt+'='+self.argDB[opt]+' is specified with --with-mpi-dir='+self.framework.argDB['with-mpi-dir']+'. However '+mpicplr+' exists and should be the prefered compiler! Suggest not specifying --'+opt+' option so that configure can use '+ mpicplr +' instead.'
                raise RuntimeError(msg)
    return

  def resetEnvCompilers(self):
    ignoreEnv = ['CC','CFLAGS','CXX','CXXFLAGS','FC','FCFLAGS','F77','FFLAGS',
                 'F90','F90FLAGS','CPP','CPPFLAGS','CXXCPP','CXXCPPFLAGS',
                 'LDFLAGS','LIBS','MPI_DIR','RM']
    for envVal in ignoreEnv:
      if envVal in os.environ:
        self.logPrintBox('***** WARNING: '+envVal+' found in environment variables - ignoring ******')
        del os.environ[envVal]
    return

  def configure(self):
    self.executeTest(self.resetEnvCompilers)
    self.executeTest(self.checkMPICompilerOverride)
    self.executeTest(self.checkVendor)
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
      self.executeTest(self.checkFortranComments)
    self.executeTest(self.checkPIC)
    self.executeTest(self.checkLargeFileIO)
    self.executeTest(self.checkArchiver)
    self.executeTest(self.checkSharedLinker)
    if Configure.isDarwin():
      self.executeTest(self.checkLinkerMac)
    self.executeTest(self.checkSharedLinkerPaths)
    self.executeTest(self.checkLibC)
    self.executeTest(self.checkDynamicLinker)
    self.executeTest(self.output)
    return

  def no_configure(self):
    if self.staticLibraries:
      self.setStaticLinker()
    return
