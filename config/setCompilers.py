from __future__ import generators
import config.base

import os

# not sure how to handle this with 'self' so its outside the class
def noCheck(command, status, output, error):
  return

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.use64BitPointers = 0
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
    help.addArgument('Compilers', '-with-cc=<prog>',  nargs.Arg(None, None, 'Specify the C compiler'))
    help.addArgument('Compilers', '-with-cxx=<prog>', nargs.Arg(None, None, 'Specify the C++ compiler'))
    help.addArgument('Compilers', '-with-fc=<prog>',  nargs.Arg(None, None, 'Specify the Fortran compiler'))

    help.addArgument('Compilers', '-with-gnu-compilers=<bool>',      nargs.ArgBool(None, 1, 'Try to use GNU compilers'))
    help.addArgument('Compilers', '-with-vendor-compilers=<vendor>', nargs.Arg(None, '', 'Try to use vendor compilers (no argument all vendors, 0 no vendors)'))
    help.addArgument('Compilers', '-with-64-bit-pointers=<bool>',    nargs.ArgBool(None, 0, 'Use 64 bit compilers and libraries'))

    help.addArgument('Compilers', '-with-large-file-io=<bool>', nargs.ArgBool(None, 0, 'Allow IO with files greater then 2 GB'))
    help.addArgument('Compilers', '-CPP=<prog>',            nargs.Arg(None, None, 'Specify the C preprocessor'))
    help.addArgument('Compilers', '-CPPFLAGS=<string>',     nargs.Arg(None, '',   'Specify the C preprocessor options'))
    help.addArgument('Compilers', '-CXXPP=<prog>',          nargs.Arg(None, None, 'Specify the C++ preprocessor'))
    help.addArgument('Compilers', '-CC=<prog>',             nargs.Arg(None, None, 'Specify the C compiler'))
    help.addArgument('Compilers', '-CFLAGS=<string>',       nargs.Arg(None, '',   'Specify the C compiler options'))
    help.addArgument('Compilers', '-CXX=<prog>',            nargs.Arg(None, None, 'Specify the C++ compiler'))
    help.addArgument('Compilers', '-CXXFLAGS=<string>',     nargs.Arg(None, '',   'Specify the C++ compiler options'))
    help.addArgument('Compilers', '-CXX_CXXFLAGS=<string>', nargs.Arg(None, '',   'Specify the C++ compiler-only options'))
    help.addArgument('Compilers', '-CXXCPPFLAGS=<string>',  nargs.Arg(None, '',   'Specify the C++ preprocessor options'))
    help.addArgument('Compilers', '-FC=<prog>',             nargs.Arg(None, None, 'Specify the Fortran compiler'))
    help.addArgument('Compilers', '-FFLAGS=<string>',       nargs.Arg(None, '',   'Specify the Fortran compiler options'))

##    help.addArgument('Compilers', '-LD=<prog>',              nargs.Arg(None, None, 'Specify the executable linker'))
##    help.addArgument('Compilers', '-CC_LD=<prog>',           nargs.Arg(None, None, 'Specify the linker for C only'))
##    help.addArgument('Compilers', '-CXX_LD=<prog>',          nargs.Arg(None, None, 'Specify the linker for C++ only'))
##    help.addArgument('Compilers', '-FC_LD=<prog>',           nargs.Arg(None, None, 'Specify the linker for Fortran only'))
    help.addArgument('Compilers', '-LD_SHARED=<prog>',       nargs.Arg(None, None, 'Specify the shared linker'))
    help.addArgument('Compilers', '-LDFLAGS=<string>',       nargs.Arg(None, '',   'Specify the linker options'))
    help.addArgument('Compilers', '-CC_LINKER_FLAGS',        nargs.Arg(None, [], 'Specify the C linker flags'))
    help.addArgument('Compilers', '-CXX_LINKER_FLAGS',       nargs.Arg(None, [], 'Specify the Cxx linker flags'))
    help.addArgument('Compilers', '-FC_LINKER_FLAGS',        nargs.Arg(None, [], 'Specify the FC linker flags'))
    help.addArgument('Compilers', '-with-ar',                nargs.Arg(None, None,   'Specify the archiver'))
    help.addArgument('Compilers', '-AR',                     nargs.Arg(None, None,   'Specify the archiver flags'))
    help.addArgument('Compilers', '-AR_FLAGS',               nargs.Arg(None, None,   'Specify the archiver flags'))
    help.addArgument('Compilers', '-with-ranlib',            nargs.Arg(None, None,   'Specify ranlib'))
    help.addArgument('Compilers', '-with-pic',               nargs.ArgBool(None, 0, 'Compile with -fPIC or equivalent flag if possible'))
    help.addArgument('Compilers', '-with-shared-ld=<prog>',  nargs.Arg(None, None, 'Specify the shared linker'))
    help.addArgument('Compilers', '-sharedLibraryFlags',     nargs.Arg(None, [], 'Specify the shared library flags'))
    help.addArgument('Compilers', '-dynamicLibraryFlags',    nargs.Arg(None, [], 'Specify the dynamic library flags'))
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
      if output.find('Unrecognised option --help passed to ld') >=0:    # NAG f95 compiler
        return 0
      if output.find('www.gnu.org') >= 0 or output.find('developer.apple.com') >= 0 or output.find('bugzilla.redhat.com') >= 0 or output.find('gcc.gnu.org') >= 0 or (output.find('gcc version')>=0 and not output.find('Intel(R)')>= 0):
        return 1
    except RuntimeError:
      pass
    return 0
  isGNU = staticmethod(isGNU)

  def isGfortran450(compiler):
    '''returns true if the compiler is gfortran450'''
    try:
      (output, error, status) = config.base.Configure.executeShellCommand(compiler+' --version')
      output = output +  error
      if output.find('GNU Fortran (GCC) 4.5.0') >=0:
        return 1
    except RuntimeError:
      pass
    return 0
  isGfortran450 = staticmethod(isGfortran450)


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
    #replace self.framework.host_cpu == 'powerpc' and self.framework.host_vendor == 'apple' and self.framework.host_os.startswith('darwin'):
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
    for language in ['C', 'Cxx', 'FC']:
      self.pushLanguage(language)
      for flagsArg in [self.getCompilerFlagsName(language), self.getCompilerFlagsName(language, 1), self.getLinkerFlagsName(language)]:
        setattr(self, flagsArg, self.argDB[flagsArg])
        self.framework.logPrint('Initialized '+flagsArg+' to '+str(getattr(self, flagsArg)))
      self.popLanguage()
    for flagsArg in ['CPPFLAGS', 'CC_LINKER_FLAGS', 'CXX_LINKER_FLAGS', 'FC_LINKER_FLAGS', 'sharedLibraryFlags', 'dynamicLibraryFlags']:
      setattr(self, flagsArg, self.argDB[flagsArg])
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
    if not self.checkLink():
      msg = 'Cannot compile/link '+language+' with '+self.getCompiler()+'.'
      self.popLanguage()
      raise RuntimeError(msg)
    if not self.framework.argDB['with-batch']:
      if not self.checkRun():
        msg = 'Cannot run executables created with '+language+'. If this machine uses a batch system \nto submit jobs you will need to configure using/configure.py with the additional option  --with-batch.\n Otherwise there is problem with the compilers. Can you compile and run code with your C/C++ (and maybe Fortran) compilers?\n'
        self.popLanguage()
        raise OSError(msg)
    self.popLanguage()
    return

  def generateCCompilerGuesses(self):
    '''Determine the C compiler using CC, then --with-cc, then MPI, then GNU, then vendors
       - Any given category can be excluded'''
    import os


    if hasattr(self, 'CC'):
      yield self.CC
    elif self.framework.argDB.has_key('with-cc'):
      if self.isWindows(self.framework.argDB['with-cc']):
        yield 'win32fe '+self.framework.argDB['with-cc']
      else:
        yield self.framework.argDB['with-cc']
      raise RuntimeError('C compiler you provided with -with-cc='+self.framework.argDB['with-cc']+' does not work')
    elif self.framework.argDB.has_key('CC'):
      if 'CC' in os.environ and os.environ['CC'] == self.framework.argDB['CC']:
        self.logPrintBox('\n*****WARNING: Using C compiler '+self.framework.argDB['CC']+' from environmental variable CC****\nAre you sure this is what you want? If not, unset that environmental variable and run configure again')
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
      raise RuntimeError('MPI compiler wrappers in '+self.framework.argDB['with-mpi-dir']+'/bin do not work. See http://www.mcs.anl.gov/petsc/petsc-as/documentation/faq.html#mpi-compilers')
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
          if self.framework.argDB['with-64-bit-pointers']:
            if Configure.isGNU(self.CC):
              self.pushLanguage('C')
              try:
                self.addCompilerFlag('-m64')
                self.use64BitPointers = 1
              except RuntimeError, e:
                self.logPrint('GNU 64-bit C compilation not working: '+str(e))
              self.popLanguage()
            elif self.vendor == 'solaris' or Configure.isSun(self.CC):
              self.pushLanguage('C')
              try:
                self.addCompilerFlag('-xarch=v9')
                self.use64BitPointers = 1
              except RuntimeError, e:
                self.logPrint('Solaris 64-bit C compilation not working: '+str(e))
              self.popLanguage()
            elif self.vendor == 'ibm' or Configure.isIBM(self.CC):
              self.pushLanguage('C')
              try:
                self.addCompilerFlag('-q64')
                self.use64BitPointers = 1
              except RuntimeError, e:
                self.logPrint('IBM 64-bit C compilation not working: '+str(e))
              self.popLanguage()
          break
      except RuntimeError, e:
        import os

        self.logPrint('Error testing C compiler: '+str(e))
        if os.path.basename(self.CC) == 'mpicc':
          self.framework.logPrint(' MPI installation '+str(self.CC)+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI.')
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
      if 'CXX' in os.environ and os.environ['CXX'] == self.framework.argDB['CXX']:
        self.logPrintBox('\n*****WARNING: Using C++ compiler '+self.framework.argDB['CXX']+' from environmental variable CXX****\nAre you sure this is what you want? If not, unset that environmental variable and run configure again')
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
            if self.framework.argDB['with-64-bit-pointers']:
              if Configure.isGNU(self.CXX):
                self.pushLanguage('C++')
                try:
                  self.addCompilerFlag('-m64')
                except RuntimeError, e:
                  self.logPrint('GNU 64-bit C++ compilation not working: '+str(e))
                self.popLanguage()
              elif self.vendor == 'solaris' or Configure.isSun(self.CXX):
                self.pushLanguage('C++')
                try:
                  self.addCompilerFlag('-xarch=v9')
                except RuntimeError, e:
                  self.logPrint('Solaris 64-bit C++ compilation not working: '+str(e))
                self.popLanguage()
              elif self.vendor == 'ibm' or Configure.isIBM(self.CXX):
                self.pushLanguage('C++')
                try:
                  self.addCompilerFlag('-q64')
                except RuntimeError, e:
                  self.logPrint('IBM 64-bit C++ compilation not working: '+str(e))
                self.popLanguage()
            break
        except RuntimeError, e:
          import os

          self.logPrint('Error testing C++ compiler: '+str(e))
          if os.path.basename(self.CXX) in ['mpicxx', 'mpiCC']:
            self.logPrint('  MPI installation '+str(self.CXX)+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI.')
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
      if 'FC' in os.environ and os.environ['FC'] == self.framework.argDB['FC']:
        self.logPrintBox('\n*****WARNING: Using Fortran compiler '+self.framework.argDB['FC']+' from environmental variable FC****\nAre you sure this is what you want? If not, unset that environmental variable and run configure again')
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
          if self.framework.argDB['with-64-bit-pointers']:
            if Configure.isGNU(self.CC):
              self.pushLanguage('FC')
              try:
                self.addCompilerFlag('-m64')
              except RuntimeError, e:
                self.logPrint('GNU 64-bit Fortran compilation not working: '+str(e))
              self.popLanguage()
            elif self.vendor == 'solaris' or Configure.isSun(self.FC):
              self.pushLanguage('FC')
              try:
                self.addCompilerFlag('-xarch=v9')
              except RuntimeError, e:
                self.logPrint('Solaris 64-bit Fortran compilation not working: '+str(e))
              self.popLanguage()
            elif self.vendor == 'ibm'  or Configure.isIBM(self.FC):
              self.pushLanguage('FC')
              try:
                self.addCompilerFlag('-q64')
              except RuntimeError, e:
                self.logPrint('IBM 64-bit Fortran compilation not working: '+str(e))
              self.popLanguage()
          break
      except RuntimeError, e:
        import os

        self.logPrint('Error testing Fortran compiler: '+str(e))
        if os.path.basename(self.FC) in ['mpif90', 'mpif77']:
         self.framework.logPrint(' MPI installation '+str(self.FC)+' is likely incorrect.\n  Use --with-mpi-dir to indicate an alternate MPI.')
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

  def addCompilerFlag(self, flag, includes = '', body = '', extraflags = '', compilerOnly = 0):
    '''Determine whether the compiler accepts the given flag, and add it if valid, otherwise throw an exception'''
    if self.checkCompilerFlag(flag+' '+extraflags, includes, body, compilerOnly):
      flagsArg = self.getCompilerFlagsArg(compilerOnly)
      setattr(self, flagsArg, getattr(self, flagsArg)+' '+flag)
      self.framework.log.write('Added '+self.language[-1]+' compiler flag '+flag+'\n')
      return
    raise RuntimeError('Bad compiler flag: '+flag)

  def checkPIC(self):
    '''Determine the PIC option for each compiler
       - There needs to be a test that checks that the functionality is actually working'''
    self.usePIC=0
    if not self.framework.argDB['with-pic'] and not self.framework.argDB['with-shared']:
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
    def checkArchive(command, status, output, error):
      if error or status:
        self.framework.logPrint('Possible ERROR while running archiver: '+output)
        if status: self.framework.logPrint('ret = '+str(status))
        if error: self.framework.logPrint('error message = {'+error+'}')
        if os.path.isfile('conf1.o'):
          os.remove('conf1.o')
        raise RuntimeError('Archiver is not functional')
      return
    def checkRanlib(command, status, output, error):
      if error or status:
        self.framework.logPrint('Possible ERROR while running ranlib: '+output)
        if status: self.framework.logPrint('ret = '+str(status))
        if error: self.framework.logPrint('error message = {'+error+'}')
        if os.path.isfile('libconf1.a'):
          os.remove('libconf1.a')
        raise RuntimeError('Ranlib is not functional with your archiver.  Try --with-ranlib=true if ranlib is unnecessary.')
      return
    oldLibs = self.LIBS
    self.pushLanguage('C')
    for (archiver, arflags, ranlib) in self.generateArchiverGuesses():
      if not self.checkCompile('', 'int foo(int a) {\n  return a+1;\n}\n\n', cleanup = 0, codeBegin = '', codeEnd = ''):
        raise RuntimeError('Compiler is not functional')
      if os.path.isfile('conf1.o'):
        os.remove('conf1.o')
      os.rename(self.compilerObj, 'conf1.o')
      if self.getExecutable(archiver, getFullPath = 1, resultName = 'AR'):
        if self.getExecutable(ranlib, getFullPath = 1, resultName = 'RANLIB'):
          arext = 'a'
          try:
            (output, error, status) = config.base.Configure.executeShellCommand(self.AR+' '+arflags+' libconf1.'+arext+' conf1.o', checkCommand = checkArchive, log = self.framework.log)
            (output, error, status) = config.base.Configure.executeShellCommand(self.RANLIB+' libconf1.'+arext, checkCommand = checkRanlib, log = self.framework.log)
          except RuntimeError, e:
            self.logPrint(str(e))
            continue
          self.LIBS = '-L. -lconf1 ' + oldLibs
          success =  self.checkLink('extern int foo(int);', '  int b = foo(1);  if (b);\n')
          os.rename('libconf1.a','libconf1.lib')
          if not success:
            arext = 'lib'
            success = self.checkLink('extern int foo(int);', '  int b = foo(1);  if (b);\n')
            os.remove('libconf1.lib')
            if success:
              break
          else:
            os.remove('libconf1.lib')
            break
    else:
      if os.path.isfile('conf1.o'):
        os.remove('conf1.o')
      self.LIBS = oldLibs
      self.popLanguage()
      raise RuntimeError('Could not find a suitable archiver.  Use --with-ar to specify an archiver.')
    self.AR_FLAGS      = arflags
    self.AR_LIB_SUFFIX = arext
    self.framework.addMakeMacro('AR_FLAGS', self.AR_FLAGS)
    self.addMakeMacro('AR_LIB_SUFFIX', self.AR_LIB_SUFFIX)
    os.remove('conf1.o')
    self.LIBS = oldLibs
    self.popLanguage()
    return

  def setStaticLinker(self):
    language = self.language[-1]
    return self.framework.setSharedLinkerObject(language, self.framework.getLanguageModule(language).StaticLinker(self.framework.argDB))

  def generateSharedLinkerGuesses(self):
    if not self.framework.argDB['with-pic'] and not self.framework.argDB['with-shared']:
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
    # C compiler default
    yield (self.CC, ['-shared'], 'so')
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
          oldLibs = self.LIBS
          self.LIBS += ' -L. -lconftest'
          if self.checkLink(includes = 'int foo(void);', body = 'int ret = foo();\nif(ret);'):
            os.remove('libconftest.'+self.sharedLibraryExt)
            self.LIBS = oldLibs
            self.sharedLibraries = 1
            self.logPrint('Using shared linker '+self.sharedLinker+' with flags '+str(self.sharedLibraryFlags)+' and library extension '+self.sharedLibraryExt)
            break
          self.LIBS = oldLibs
          os.remove('libconftest.'+self.sharedLibraryExt)
        if os.path.isfile(self.linkerObj): os.remove(self.linkerObj)
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
    else:
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
    langMap = {'C':'CC','FC':'FC','Cxx':'CXX'}
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
    '''Check that the linker can produce dynamic libraries'''
    self.dynamicLibraries = 0
    if not self.headers.check('dlfcn.h'):
      self.logPrint('Dynamic libraries disabled since dlfcn.h was missing')
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
        if self.checkLink(includes = 'int '+testMethod+'(void) {return 0;}\n', codeBegin = '', codeEnd = '', cleanup = 0, shared = 'dynamic'):
          code = '''
void *handle = dlopen("./libconftest.so", 0);
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
'''
          if self.checkLink(includes = '#include<dlfcn.h>', body = code):
            os.remove('libconftest.'+self.dynamicLibraryExt)
            self.dynamicLibraries = 1
            self.logPrint('Using dynamic linker '+self.dynamicLinker+' with flags '+str(self.dynamicLibraryFlags)+' and library extension '+self.dynamicLibraryExt)
            break
          os.remove('libconftest.'+self.dynamicLibraryExt)
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
    if hasattr(self, 'CXX'):
      self.addSubstitution('CXX', self.CXX)
      self.addSubstitution('CXX_CXXFLAGS', self.CXX_CXXFLAGS)
      self.addSubstitution('CXXFLAGS', self.CXXFLAGS)
      self.addSubstitution('CXX_LINKER_SLFLAG', self.CxxSharedLinkerFlag)
    else:
      self.addSubstitution('CXX', '')
    if hasattr(self, 'CXXCPP'):
      self.addSubstitution('CXXCPP', self.CXXCPP)
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
    
    opts = ['with-cc','with-fc','with-cxx','CC','FC','CXX']
    optsMatch = []
    if 'with-mpi-dir' in self.argDB:
      for opt in opts:
        if (opt in self.argDB  and self.argDB[opt] != '0'):
          optsMatch.append(opt)
    if optsMatch:
      mesg = '''\
Warning: [with-mpi-dir] option is used along with options: ''' + str(optsMatch) + '''
This prevents configure from picking up MPI compilers from specified mpi-dir.

Suggest using *only* [with-mpi-dir] option - and no other compiler option.
This way - mpi compilers from '''+self.argDB['with-mpi-dir']+ ''' are used.'''
      self.logPrintBox(mesg)
    return

  def configure(self):
    self.executeTest(self.checkMPICompilerOverride)
    self.executeTest(self.checkVendor)
    self.executeTest(self.checkInitialFlags)
    self.executeTest(self.checkCCompiler)
    self.executeTest(self.checkCPreprocessor)
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
