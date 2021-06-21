import config.base

import re
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.version = {}
    self.rejected = {}
    self.text = ''
    return

  def __str__(self):
    return self.text

  def setupHelp(self, help):
    import nargs
    help.addArgument('Compiler Flags', '-optionsModule=<module name>', nargs.Arg(None, 'config.compilerOptions', 'The Python module used to determine compiler options and versions'))
    help.addArgument('Compiler Flags', '-with-debugging=<bool>', nargs.ArgBool(None, 1, 'Specify debugging version of libraries'))
    help.addArgument('Compiler Flags', '-C_VERSION=<string>',   nargs.Arg(None, 'Unknown', 'The version of the C compiler'))
    help.addArgument('Compiler Flags', '-CXX_VERSION=<string>', nargs.Arg(None, 'Unknown', 'The version of the C++ compiler'))
    help.addArgument('Compiler Flags', '-FC_VERSION=<string>',  nargs.Arg(None, 'Unknown', 'The version of the Fortran compiler'))
    help.addArgument('Compiler Flags', '-CUDA_VERSION=<string>',nargs.Arg(None, 'Unknown', 'The version of the CUDA compiler'))
    help.addArgument('Compiler Flags', '-HIP_VERSION=<string>',nargs.Arg(None, 'Unknown', 'The version of the HIP compiler'))
    help.addArgument('Compiler Flags', '-SYCL_VERSION=<string>',nargs.Arg(None, 'Unknown', 'The version of the SYCL compiler'))
    help.addArgument('Compiler Flags', '-COPTFLAGS=<string>',   nargs.Arg(None, None, 'Override the debugging/optimization flags for the C compiler'))
    help.addArgument('Compiler Flags', '-CXXOPTFLAGS=<string>', nargs.Arg(None, None, 'Override the debugging/optimization flags for the C++ compiler'))
    help.addArgument('Compiler Flags', '-FOPTFLAGS=<string>',   nargs.Arg(None, None, 'Override the debugging/optimization flags for the Fortran compiler'))
    help.addArgument('Compiler Flags', '-CUDAOPTFLAGS=<string>',   nargs.Arg(None, None, 'Override the debugging/optimization flags for the CUDA compiler'))
    help.addArgument('Compiler Flags', '-HIPOPTFLAGS=<string>',   nargs.Arg(None, None, 'Override the debugging/optimization flags for the HIP compiler'))
    help.addArgument('Compiler Flags', '-SYCLOPTFLAGS=<string>',   nargs.Arg(None, None, 'Override the debugging/optimization flags for the SYCL compiler'))
    # not sure where to put this, currently gcov is handled in ../compilerOptions.py
    # not sure where to put this, currently gcov is handled in ../compilerOptions.py
    help.addArgument('Compiler Flags', '-with-gcov=<bool>', nargs.ArgBool(None, 0, 'Specify that GNUs coverage tool gcov is used'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers', self)
    return

  def getOptionalFlagsName(self, language, compilerOnly = 0):
    if language == 'C':
      flagsArg = 'COPTFLAGS'
    elif language == 'Cxx':
      if compilerOnly:
        flagsArg = 'CXX_CXXOPTFLAGS'
      else:
        flagsArg = 'CXXOPTFLAGS'
    elif language == 'FC':
      flagsArg = 'FOPTFLAGS'
    elif language == 'CUDA':
      flagsArg = 'CUDAOPTFLAGS'
    elif language == 'HIP':
      flagsArg = 'HIPOPTFLAGS'
    elif language == 'SYCL':
      flagsArg = 'SYCLOPTFLAGS'
    else:
      raise RuntimeError('Unknown language: '+language)
    return flagsArg

  def hasOptFlags(self,flags):
    for flag in flags.split():
      if flag.startswith('-g') or flag.startswith('-O') or flag in ['-fast']:
        return 1
    return 0

  def getOptionsObject(self):
    '''Get a configure object which will return default options for each compiler'''
    options = None
    try:
      mod     = __import__(self.argDB['optionsModule'], locals(), globals(), ['CompilerOptions'])
      options = mod.CompilerOptions(self.framework)
      options.setup()
    except ImportError:
      self.logPrint('ERROR: Failed to load user options module '+str(self.argDB['optionsModule']))
    return options

  def configureCompilerFlags(self):
    '''Get the default compiler flags'''
    self.debugging = self.argDB['with-debugging']
    bopts = ['']
    if self.debugging:
      bopts.append('g')
    else:
      bopts.append('O')

    # According to gcc doc, gcov does not require -g, so we do it alone
    if self.argDB['with-gcov']:
      bopts.append('gcov')
      self.addDefine('USE_GCOV', 1)

    options = self.getOptionsObject()
    if not options:
      return
    options.saveLog()
    for language, compiler in [('C', 'CC'), ('Cxx', 'CXX'), ('FC', 'FC'), ('CUDA', 'CUDAC'), ('HIP', 'HIPC'), ('SYCL', 'SYCLCXX')]:
      if not hasattr(self.setCompilers, compiler):
        continue
      self.setCompilers.pushLanguage(language)
      flagsName = config.base.Configure.getCompilerFlagsName(language)
      try:
        self.version[language] = self.argDB[language.upper()+'_VERSION']
        if self.version[language] == 'Unknown':
          self.version[language] = options.getCompilerVersion(language, self.setCompilers.getCompiler())
      except RuntimeError:
        pass
      self.rejected[language] = []
      for bopt in bopts:
        userflags = 0
        if bopt in ['g','O'] and self.getOptionalFlagsName(language) in self.argDB: # check --COPTFLAGS etc
          # treat user supplied options as single option - as it could include options separated by spaces '-tp k8-64'
          flags = [self.argDB[self.getOptionalFlagsName(language)]]
          userflags = 1
        elif bopt in ['g','O'] and self.hasOptFlags(getattr(self.setCompilers,flagsName)): # check --CFLAGS etc
          self.logPrint('Optimization options found in '+flagsName+ '. Skipping setting defaults')
          flags = []
        elif bopt == '' and flagsName in self.argDB:
          self.logPrint('Ignoring default options which were overridden using --'+flagsName+ ' ' + self.argDB[flagsName])
          flags = []
        else:
          flags = options.getCompilerFlags(language, self.setCompilers.getCompiler(), bopt)

        for testFlag in flags:
          if isinstance(testFlag,tuple):
            testFlag = ' '.join(testFlag)
          try:
            self.logPrint('Trying '+language+' compiler flag '+testFlag)
            self.setCompilers.addCompilerFlag(testFlag)
          except RuntimeError:
            if userflags:
              raise RuntimeError('User provided flags for language '+language+' with '+self.getOptionalFlagsName(language)+': '+self.argDB[self.getOptionalFlagsName(language)]+' are not correct for the compiler')
            self.logPrint('Rejected '+language+' compiler flag '+testFlag)
            self.rejected[language].append(testFlag)
      self.setCompilers.popLanguage()
    return

  def outputCompilerMacros(self):
    '''Cannot use the regular outputCompile because it swallows the output into the -o file'''
    command = self.getCompilerCmd()
    if self.compilerDefines: self.framework.outputHeader(self.compilerDefines)
    self.framework.outputCHeader(self.compilerFixes)
    f = open(self.compilerSource, 'w')
    f.close()
    try:
      command = command + ' -E -dM '
      start = command.find('-o ')
      end = start + 3 + command[start+3:].find(' ')
      command = command[:start-1]+command[end:]
      (out, err, ret) = Configure.executeShellCommand(command, log = self.log)
      if out.find('__AVX2__') > -1 and out.find('__FMA__') > -1:
        self.text = self.text + 'Intel instruction sets utilizable by compiler:\n'
        self.text = self.text + '  AVX2\n'
      if out.find('__AVX512__') > -1:
        self.text = self.text + '  AVX512\n'
    except:
      pass
    for filename in [self.compilerDefines, self.compilerFixes, self.compilerSource, self.compilerObj]:
      if os.path.isfile(filename): os.remove(filename)

  def checkCompilerMacros(self):
    '''Save the list of CPP macros defined by the C and C++ compiler, does not work for all compilers'''
    '''The values will depends on the flags passed to the compiler'''
    self.outputCompilerMacros()
    if hasattr(self.setCompilers, 'CXX'):
      self.pushLanguage('Cxx')
      self.outputCompilerMacros()
    return

  def checkIntelHardwareSupport(self):
    '''Use Linux/MacOS commands to determine what operations the hardware supports'''
    try:
      (out, err, ret) = Configure.executeShellCommand('lscpu', log = self.log)
    except:
      try:
        (out, err, ret) = Configure.executeShellCommand('sysctl -a', log = self.log)
        if out.find('hw.optional.avx2_0: 1') > -1 and out.find('hw.optional.fma: 1') > -1:
          self.text = self.text + 'Intel instruction sets found on CPU:\n'
          self.text = self.text + '  AVX2\n'
        if out.find('hw.optional.avx512f: 1') > -1:
          self.text = self.text + '  AVX512\n'
      except:
        pass
    return

  def configure(self):
    self.executeTest(self.configureCompilerFlags)
    self.executeTest(self.checkIntelHardwareSupport)
    self.executeTest(self.checkCompilerMacros)
    return
